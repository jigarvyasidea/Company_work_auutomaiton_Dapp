import operator
import tempfile
import os
import json
import boto3
from io import BytesIO
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END



# Import existing components
from upload.embeddings import EmbeddingGenerator
from upload.llm_question_generator import LLMQuestionGenerator
from upload.store_questions_answers import StoreQuestionsAnswers
from upload.vector_store import VectorStorePGVector
from upload.validation_api import fetch_questions_from_db, validate_locally, validate_with_groq

# Initialize components (these will be passed to nodes or initialized within them)
embedding_generator = EmbeddingGenerator()
question_generator = LLMQuestionGenerator()
store_qa = StoreQuestionsAnswers()
vector_store = VectorStorePGVector()
s3 = boto3.client("s3")

class GraphState(TypedDict):
    book_id: str
    user_id: str
    file_path: Optional[str]
    s3_url: Optional[str]
    total_questions: Optional[int]
    questions_answers: Optional[List[Dict[str, Any]]]
    row_id: Optional[str]
    created_on: Optional[str]
    error: Optional[str]
    chapters_data: Optional[List[Dict[str, Any]]]
    user_answers_list: Optional[List[Dict[str, Any]]]
    validated_results: Optional[List[Dict[str, Any]]]
    api_response: Optional[Dict[str, Any]]

# --- Nodes for Question Generation Workflow ---

def prepare_input(state: GraphState) -> GraphState:
    print("---PREPARE INPUT---")
    book_id = state.get("book_id")
    user_id = state.get("user_id")
    file_content = state.get("file_content") # This will be passed from FastAPI
    s3_url = state.get("s3_url")
    total_questions = state.get("total_questions")

    if not file_content and not s3_url:
        return {**state, "error": "Either an uploaded file or an S3 URL is required"}
    if file_content and s3_url:
        return {**state, "error": "Provide either an uploaded file or an S3 URL, not both"}

    temp_file_path = None
    try:
        if s3_url:
            if not s3_url.startswith("s3://"):
                raise ValueError("Invalid S3 URL format")
            s3_parts = s3_url.replace("s3://", "").split("/", 1)
            if len(s3_parts) < 2:
                raise ValueError("Invalid S3 URL format")
            bucket_name, file_key = s3_parts
            file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
            file_content = BytesIO(file_obj["Body"].read()).read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        return {**state, "file_path": temp_file_path}
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {**state, "error": f"Error in prepare_input: {str(e)}"}


def check_embeddings(state: GraphState) -> GraphState:
    print("---CHECK EMBEDDINGS---")
    book_id = state.get("book_id")
    if not book_id:
        return {**state, "error": "book_id is missing for check_embeddings"}
    
    embeddings_exist = vector_store.check_if_record_exist(book_id)
    return {**state, "embeddings_exist": embeddings_exist}


def generate_embeddings(state: GraphState) -> GraphState:
    print("---GENERATE EMBEDDINGS---")
    file_path = state.get("file_path")
    book_id = state.get("book_id")

    if not file_path or not book_id:
        return {**state, "error": "file_path or book_id is missing for generate_embeddings"}

    try:
        embedding_generator.create_embeddings(file_path, book_id)
        return state
    except Exception as e:
        return {**state, "error": f"Error generating embeddings: {str(e)}"}
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path) # Clean up temporary file

def generate_questions(state: GraphState) -> GraphState:
    print("---GENERATE QUESTIONS---")
    book_id = state.get("book_id")
    user_id = state.get("user_id")
    total_questions = state.get("total_questions")

    if not book_id or not user_id:
        return {**state, "error": "book_id or user_id is missing for generate_questions"}

    try:
        if total_questions:
            questions_answers = question_generator.generate_questions_for_book(book_id, user_id, total_questions=total_questions)
        else:
            questions_answers = question_generator.generate_questions_for_book(book_id, user_id)
        return {**state, "questions_answers": questions_answers}
    except Exception as e:
        return {**state, "error": f"Error generating questions: {str(e)}"}


def store_questions(state: GraphState) -> GraphState:
    print("---STORE QUESTIONS---")
    user_id = state.get("user_id")
    book_id = state.get("book_id")
    questions_answers = state.get("questions_answers")

    if not user_id or not book_id or not questions_answers:
        return {**state, "error": "Missing data for storing questions"}

    try:
        store_qa.store_questions_answers(user_id, book_id, questions_answers)
        return state
    except Exception as e:
        return {**state, "error": f"Error storing questions: {str(e)}"}


def fetch_stored_questions(state: GraphState) -> GraphState:
    print("---FETCH STORED QUESTIONS---")
    user_id = state.get("user_id")
    book_id = state.get("book_id")

    if not user_id or not book_id:
        return {**state, "error": "Missing user_id or book_id for fetching questions"}

    try:
        stored_data = store_qa.fetch_questions_answers(user_id, book_id)
        if not stored_data or "chapters" not in stored_data:
            return {**state, "error": "No questions found in DB"}

        return {**state,
                "questions_answers": stored_data["chapters"],
                "row_id": stored_data["row_id"],
                "created_on": stored_data["created_on"]}
    except Exception as e:
        return {**state, "error": f"Error fetching stored questions: {str(e)}"}


def format_api_response(state: GraphState) -> GraphState:
    print("---FORMAT API RESPONSE---")
    user_id = state.get("user_id")
    book_id = state.get("book_id")
    questions_answers = state.get("questions_answers")
    row_id = state.get("row_id")
    created_on = state.get("created_on")

    if not all([user_id, book_id, questions_answers, row_id, created_on]):
        return {**state, "error": "Missing data for formatting API response"}

    formatted_questions = []

    def determine_question_type(qa):
        if "options" in qa:
            return "MCQ"
        elif "answer" in qa and isinstance(qa["answer"], bool):
            return "True/False"
        elif "question" in qa and "___" in qa["question"]:
            return "Fill_in_the_blanks"
        else:
            return "Text"

    for chapter in questions_answers:
        chapter_name = chapter["chapter_name"]
        chapter_number = chapter.get("chapter_number", None)

        for qa in chapter["questions_answers"]:
            question_data = {
                "question": qa["question"],
                "question_type": determine_question_type(qa),
                "chapter_name": chapter_name,
                "chapter_number": chapter_number,
                "question_id": qa["question_id"]
            }

            if qa.get("options"):
                question_data["options"] = {
                    "option1": qa["options"][0],
                    "option2": qa["options"][1],
                    "option3": qa["options"][2],
                    "option4": qa["options"][3]
                }

            formatted_questions.append(question_data)
            
            if len(formatted_questions) >= 40:
                break
        if len(formatted_questions) >= 40:
            break

    api_response = {
        "user_id": user_id,
        "book_id": book_id,
        "created_on": created_on,
        "status": "success",
        "row_id": row_id,
        "list": formatted_questions[:40]
    }
    return {**state, "api_response": api_response}

# --- Nodes for Answer Validation Workflow ---

def fetch_questions_for_validation(state: GraphState) -> GraphState:
    print("---FETCH QUESTIONS FOR VALIDATION---")
    row_id = state.get("row_id")
    if not row_id:
        return {**state, "error": "row_id is missing for validation fetch"}
    
    try:
        chapters_data = fetch_questions_from_db(row_id)
        if not chapters_data:
            return {**state, "error": "No data found for the provided row_id for validation."}
        return {**state, "chapters_data": chapters_data}
    except Exception as e:
        return {**state, "error": f"Error fetching questions for validation: {str(e)}"}


def validate_answers(state: GraphState) -> GraphState:
    print("---VALIDATE ANSWERS---")
    user_answers_list = state.get("user_answers_list")
    chapters_data = state.get("chapters_data")

    if not user_answers_list or not chapters_data:
        return {**state, "error": "Missing user_answers_list or chapters_data for validation"}

    question_map = {}
    for chapter in chapters_data:
        if isinstance(chapter, dict):
            chapter_name = chapter.get("chapter_name", "Unknown Chapter")
            chapter_number = chapter.get("chapter_number", 0)
            questions_answers = chapter.get("questions_answers", [])

            if isinstance(questions_answers, str):
                try:
                    questions_answers = json.loads(questions_answers)
                except json.JSONDecodeError:
                    questions_answers = []

            if isinstance(questions_answers, list):
                for qa in questions_answers:
                    if isinstance(qa, dict):
                        question_id = str(qa.get("question_id"))
                        question_type = qa.get("question_type", "").lower()
                        options = qa.get("options", [])
                        correct_answer = qa.get("answer", "")

                        correct_index = None
                        if question_type == "mcq" and isinstance(options, list):
                            for idx, opt in enumerate(options):
                                if str(opt).strip().lower() == str(correct_answer).strip().lower():
                                    correct_index = idx
                                    break

                        question_map[question_id] = {
                            "question": qa.get("question", ""),
                            "question_type": qa.get("question_type", ""),
                            "chapter_name": chapter_name,
                            "chapter_id": chapter_number,
                            "correct_answer": correct_index if question_type == "mcq" else correct_answer,
                            "options": options if question_type == "mcq" else {}
                        }

    validated_results = []
    for user_answer in user_answers_list:
        question_id = str(user_answer["question_id"])
        question_data = question_map.get(question_id)

        if not question_data:
            return {**state, "error": f"Invalid question_id: {question_id}"}

        user_answer_text = user_answer.get("user_answer", "")
        qtype = question_data["question_type"].lower()
        correct_ans = question_data["correct_answer"]
        options = question_data.get("options", [])

        if qtype == "mcq":
            is_correct = validate_locally(qtype, user_answer_text, correct_ans, options)
            score = 10 if is_correct else 0
            is_correct = "true" if is_correct else "false"
        else:
            score, is_correct = validate_with_groq(
                question_data["question"], user_answer_text, correct_ans
            )

        validated_results.append({
            "question_id": question_id,
            "question": question_data["question"],
            "question_type": question_data["question_type"],
            "chapter_name": question_data["chapter_name"],
            "chapter_id": question_data["chapter_id"],
            "correct_answer": correct_ans,
            "user_answer": user_answer_text,
            "options": options if qtype == "mcq" else {},
            "score": score,
            "is_correct": is_correct
        })
    return {**state, "validated_results": validated_results}


def format_validation_response(state: GraphState) -> GraphState:
    print("---FORMAT VALIDATION RESPONSE---")
    user_id = state.get("user_id")
    validated_results = state.get("validated_results")

    if not user_id or not validated_results:
        return {**state, "error": "Missing user_id or validated_results for formatting validation response"}

    import datetime
    api_response = {
        "user_id": user_id,
        "created_on": datetime.datetime.utcnow().isoformat(),
        "status": "success",
        "list": validated_results
    }
    return {**state, "api_response": api_response}

# --- Conditional Edges ---

def decide_to_generate_embeddings(state: GraphState):
    print("---DECIDE TO GENERATE EMBEDDINGS---")
    if state.get("error"):
        return "end_error"
    if state.get("embeddings_exist"):
        return "fetch_stored_questions"
    else:
        return "generate_embeddings"


def decide_end_or_continue(state: GraphState):
    print("---DECIDE END OR CONTINUE---")
    if state.get("error"):
        return "end_error"
    return "format_api_response"

# --- Build the Graph ---

def build_question_generation_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("prepare_input", prepare_input)
    workflow.add_node("check_embeddings", check_embeddings)
    workflow.add_node("generate_embeddings", generate_embeddings)
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("store_questions", store_questions)
    workflow.add_node("fetch_stored_questions", fetch_stored_questions)
    workflow.add_node("format_api_response", format_api_response)

    # Set entry point
    workflow.set_entry_point("prepare_input")

    # Add edges
    workflow.add_edge("prepare_input", "check_embeddings")
    workflow.add_conditional_edges(
        "check_embeddings",
        decide_to_generate_embeddings,
        {
            "fetch_stored_questions": "fetch_stored_questions",
            "generate_embeddings": "generate_embeddings",
            "end_error": END # Handle errors early
        },
    )
    workflow.add_edge("generate_embeddings", "generate_questions")
    workflow.add_edge("generate_questions", "store_questions")
    workflow.add_edge("store_questions", "fetch_stored_questions") # Fetch again to get row_id and created_on
    workflow.add_edge("fetch_stored_questions", "format_api_response")
    workflow.add_edge("format_api_response", END)

    return workflow.compile()

def build_validation_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("fetch_questions_for_validation", fetch_questions_for_validation)
    workflow.add_node("validate_answers", validate_answers)
    workflow.add_node("format_validation_response", format_validation_response)

    # Set entry point
    workflow.set_entry_point("fetch_questions_for_validation")

    # Add edges
    workflow.add_edge("fetch_questions_for_validation", "validate_answers")
    workflow.add_edge("validate_answers", "format_validation_response")
    workflow.add_edge("format_validation_response", END)

    return workflow.compile()

# Compile the graphs
question_generation_app = build_question_generation_graph()
validation_app = build_validation_graph()


