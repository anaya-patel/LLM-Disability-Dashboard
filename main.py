from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import uvicorn
from datetime import datetime

# Import services
from services.openai_service import generate_question, test_openai_connection
from services.database_service import (
    save_user_data, 
    save_feedback,
    get_student_history,
    create_student
)

app = FastAPI(
    title="Educational Dashboard API",
    description="API for an educational dashboard using AI to generate and analyze math questions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class Student(BaseModel):
    name: str
    grade: str
    age: int

# Updated Pydantic model to match your requested format
class QuestionRequest(BaseModel):
    Name: Optional[str] = ""
    Age: Optional[str] = ""
    Grade: Optional[int] = -1
    Subject: Optional[str] = ""
    "Given-questions": Optional[int] = -1
    "Correct-answered": Optional[int] = -1
    "Known-disability": Optional[bool] = False
    "Given-question": Optional[str] = ""
    "Mistake-made": Optional[str] = ""
    "Time-taken": Optional[str] = ""
    "Additional-observation": Optional[str] = ""

# Updated response model to match your requested format
class QuestionResponse(BaseModel):
    Question: str
    Mistakes: List[str]
    Reasons: List[str]
    approaches: List[str]

class SuccessResponse(BaseModel):
    success: bool = True
    id: Optional[str] = None
    data: Optional[Any] = None

@app.post("/api/student", response_model=SuccessResponse)
async def create_student_route(student: Student):
    """Create a new student record"""
    try:
        result = await create_student(student.dict())
        return SuccessResponse(success=True, id=result["id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create student: {str(e)}")

@app.post("/api/generate-question", response_model=QuestionResponse)
async def generate_question_route(request: QuestionRequest):
    """Generate a personalized question based on student information"""
    try:
        # Convert Pydantic model to dict
        student_data = request.dict(by_alias=True)
        
        # Generate question
        question_data = await generate_question(student_data)
        
        # Save session data (optional)
        await save_user_data({
            "studentInfo": student_data,
            "generatedQuestion": question_data,
            "timestamp": datetime.now().isoformat(),
            "sessionType": "question_generation"
        })
        
        return question_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate question: {str(e)}")

@app.get("/api/test-openai", response_model=SuccessResponse)
async def test_openai_connection_route():
    """Simple endpoint to test OpenAI connection"""
    try:
        result = await test_openai_connection()
        return SuccessResponse(success=result["success"], data=result["message"] if "message" in result else result["error"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to OpenAI: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)