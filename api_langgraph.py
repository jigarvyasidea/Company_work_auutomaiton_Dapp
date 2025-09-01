from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from langgraph_workflow import question_generation_app, validation_app, GraphState
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

@app.post("/generate-questions")
async def generate_questions(
    book_id: str = Form(...),
    user_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    total_questions: Optional[int] = Form(None),
    s3_url: Optional[str] = Form(None)
):
    file_content = None
    if file:
        file_content = await file.read()

    initial_state = GraphState(
        book_id=book_id,
        user_id=user_id,
        file_content=file_content,
        s3_url=s3_url,
        total_questions=total_questions
    )

    try:
        # Run the LangGraph workflow
        final_state = question_generation_app.invoke(initial_state)

        if final_state.get("error"):
            raise HTTPException(status_code=400, detail=final_state["error"])

        return final_state.get("api_response")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/validate_answers")
async def validate_answers(payload: dict):
    user_id = payload.get("user_id")
    row_id = payload.get("row_id")
    user_answers_list = payload.get("list", [])

    if not user_id or not row_id or not user_answers_list:
        raise HTTPException(status_code=400, detail="Missing required fields.")

    initial_state = GraphState(
        user_id=user_id,
        row_id=row_id,
        user_answers_list=user_answers_list
    )

    try:
        # Run the LangGraph workflow
        final_state = validation_app.invoke(initial_state)

        if final_state.get("error"):
            raise HTTPException(status_code=400, detail=final_state["error"])

        return final_state.get("api_response")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")





