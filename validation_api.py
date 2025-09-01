import os
import json
import datetime
from fastapi import FastAPI, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from upload.prompts import VALIDATION_PROMPT

load_dotenv()
app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DBNAME = os.getenv("DB_NAME")
DBUSER = os.getenv("DB_USER")
DBPW = os.getenv("DB_PASSWORD")
DBHOST = os.getenv("DB_HOST")
DBPORT = os.getenv("DB_PORT")


def fetch_questions_from_db(row_id):
    query = "SELECT questions_answers FROM questions_and_answers WHERE row_id = %s;"
    try:
        conn = psycopg2.connect(
            dbname=DBNAME, user=DBUSER, password=DBPW, host=DBHOST, port=DBPORT
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (row_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result:
            questions_data = result["questions_answers"]
            if isinstance(questions_data, str):
                questions_data = json.loads(questions_data)
            return questions_data.get("chapters", [])
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


def validate_locally(question_type, user_answer, correct_answer_index, options):
    question_type = question_type.lower()
    user_answer_clean = str(user_answer).strip().lower()

    if question_type == "mcq":
        try:
            user_index = int(user_answer_clean)
            return user_index == correct_answer_index
        except (ValueError, TypeError):
            return False
    elif question_type == "true/false":
        return user_answer_clean == str(correct_answer_index).strip().lower()
    return None


def validate_with_groq(question_text, user_answer, correct_answer):
    try:
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")

        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

        llm_input = {
            "question": question_text,
            "user_answer": user_answer,
            "correct_answer": correct_answer
        }

        response = llm.invoke([
            {"role": "system", "content": VALIDATION_PROMPT},
            {"role": "user", "content": json.dumps(llm_input)}
        ])

        output = json.loads(response.content)

        score = output.get("score", 0)
        is_correct = "true" if str(output.get("is_correct")).lower() == "true" else "false"

        return score, is_correct

    except (json.JSONDecodeError, AttributeError):
        return 0, "false"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM validation error: {str(e)}")


@app.post("/validate_answers")
async def validate_answers(payload: dict):
    user_id = payload.get("user_id")
    row_id = payload.get("row_id")
    user_answers_list = payload.get("list", [])

    if not user_id or not row_id or not user_answers_list:
        raise HTTPException(status_code=400, detail="Missing required fields.")

    chapters_data = fetch_questions_from_db(row_id)
    if not chapters_data:
        raise HTTPException(status_code=404, detail="No data found for the provided row_id.")

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

                        # For MCQ questions, find the index of the correct answer
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
            raise HTTPException(status_code=400, detail=f"Invalid question_id: {question_id}")

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

    return {
        "user_id": user_id,
        "created_on": datetime.datetime.utcnow().isoformat(),
        "status": "success",
        "list": validated_results
    }
