import json
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from upload.vector_store import VectorStorePGVector
from upload.prompts import QUESTION_GENERATION_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class LLMQuestionGenerator:
    def __init__(self, collection_name: str = "book_embeddings"):
        self.vector_store = VectorStorePGVector(collection_name)
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    def generate_questions_for_book(self, book_id: str, user_id: str) -> List[Dict[str, Any]]:
        questions_answers = []
        
        # Retrieve all chapters in the order they appear in the book
        retrieved_chapters = self.vector_store.retrieve_relevant_chapter_chunks(book_id, "", k=50)
        
        # Extract chapter names and ensure they remain in the order found in the book
        chapter_names = list(dict.fromkeys(chunk["chapter_name"] for chunk in retrieved_chapters))
        # print("retrieved chapters from vector store", retrieved_chapters)
        
        if not chapter_names:
            return []

        for chapter_number, chapter_name in enumerate(chapter_names, start=1):
            chapter_chunks = self.vector_store.retrieve_relevant_chapter_chunks(book_id, chapter_name, k=10)
            print("length of chapters", len(chapter_chunks))
            combined_text = " ".join(chunk["text"] for chunk in chapter_chunks)[:6000]

            # Generate questions using LLM
            prompt = QUESTION_GENERATION_PROMPT.format(text=combined_text, chapter_name=chapter_name)

            response_text = ""

            try:
                response = self.llm.invoke(prompt)
                response_text = response.content if hasattr(response, "content") else str(response)

                if not response_text.strip():
                    print(f" Warning: Empty response from LLM for chapter: {chapter_name}")
                    continue

                # print(f"Raw LLM Response for {chapter_name}:\n{response_text}\n")

                # Extract JSON from response
                json_start = response_text.find("{")  
                json_end = response_text.rfind("}")
                if json_start == -1 or json_end == -1:
                    raise ValueError(f"No valid JSON found in response: {response_text}")

                response_text = response_text[json_start : json_end + 1]

                # Parse JSON response
                qa_data = json.loads(response_text)

                # Ensure expected keys exist
                if "chapter_name" not in qa_data or "questions_answers" not in qa_data:
                    raise KeyError("Missing expected keys ('chapter_name', 'questions_answers') in response JSON.")

            except json.JSONDecodeError as e:
                print(f" Error parsing JSON for chapter: {chapter_name} - {str(e)}\nResponse: {response_text}\n")
                continue
            except Exception as e:
                print(f" Unexpected error for chapter: {chapter_name} - {str(e)}\nResponse: {response_text}\n")
                continue

            formatted_qa = {
                "user_id": user_id,
                "book_id": book_id,
                "chapter_name": qa_data["chapter_name"], 
                "chapter_number": chapter_number,
                "questions_answers": qa_data["questions_answers"]
            }

            questions_answers.append(formatted_qa)

        return questions_answers
