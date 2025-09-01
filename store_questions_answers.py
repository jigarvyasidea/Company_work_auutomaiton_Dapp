import os
import json
import psycopg2
import random
import string
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DBNAME = os.getenv("DB_NAME")
DBUSER = os.getenv("DB_USER")
DBPW = os.getenv("DB_PASSWORD")
DBHOST = os.getenv("DB_HOST")
DBPORT = os.getenv("DB_PORT")

class StoreQuestionsAnswers:
    def __init__(self):
        """Initializes the database connection."""
        self.connection = psycopg2.connect(
            dbname=DBNAME, user=DBUSER, password=DBPW, host=DBHOST, port=DBPORT
        )
        self.cursor = self.connection.cursor()

    def generate_row_id(self):
        """Generates a random 16-character alphanumeric row_id."""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))

    def store_questions_answers(self, user_id: str, book_id: str, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stores all generated questions and answers as a single JSON in the database."""

        if not qa_data:
            return {"status": "error", "message": "No questions generated"}

        row_id = self.generate_row_id() 
        created_on = datetime.now().isoformat()  

        # Ensure each question has a unique ID and chapter number
        question_id_counter = 100 
        for chapter_number, chapter in enumerate(qa_data, start=1):
            chapter["chapter_number"] = chapter_number 

            for qa in chapter["questions_answers"]:
                qa["question_id"] = question_id_counter 
                question_id_counter += 1

        # Store combined JSON
        combined_qa_json = json.dumps({"chapters": qa_data})

        self.cursor.execute(
            """
            INSERT INTO questions_and_answers (user_id, book_id, questions_answers, time, row_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, book_id) DO UPDATE
            SET questions_answers = EXCLUDED.questions_answers, time = EXCLUDED.time, row_id = EXCLUDED.row_id;
            """,
            (user_id, book_id, combined_qa_json, created_on, row_id),
        )

        # self.cursor.execute(
        #     """
        #     INSERT INTO questions_and_answers (user_id, book_id, questions_answers, time, row_id)
        #     VALUES (%s, %s, %s, %s, %s);
        #     """,
        #     (user_id, book_id, combined_qa_json, created_on, row_id),
        # )
        
        self.connection.commit()
        return {"status": "success", "message": "Questions stored successfully", "row_id": row_id}

    def fetch_questions_answers(self, user_id: str, book_id: str) -> Dict[str, Any]:
            """Fetches stored questions and answers for a specific user and book."""
            self.cursor.execute(
                """
                SELECT questions_answers, row_id, time FROM questions_and_answers
                WHERE user_id = %s AND book_id = %s;
                """,
                (user_id, book_id),
            )
            result = self.cursor.fetchone()

            if result:
                # ✅ Fix: Only decode JSON if it's a string
                if isinstance(result[0], str):
                    questions_answers = json.loads(result[0])
                else:
                    questions_answers = result[0]  

                row_id = result[1]
                created_on = result[2]
                return {
                    "chapters": questions_answers["chapters"],
                    "row_id": row_id,
                    "created_on": created_on
                }

            return {"chapters": [], "row_id": None, "created_on": None}



    def close_connection(self):
        self.cursor.close()
        self.connection.close()
