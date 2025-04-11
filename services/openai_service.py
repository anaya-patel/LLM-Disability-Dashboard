import os
from typing import Dict, List, Any, Optional
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client with the latest client version
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_question(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a personalized question based on student information
    
    Args:
        student_data: Dictionary containing student information with the following structure:
            {
                "Name": student name,
                "Age": student age,
                "Grade": student grade level,
                "Subject": subject area,
                "Given-questions": number of questions given previously,
                "Correct-answered": number of correctly answered questions,
                "Known-disability": boolean indicating if student has known disability,
                "Given-question": previously given question,
                "Mistake-made": description of mistakes made,
                "Time-taken": time taken to answer,
                "Additional-observation": any additional observations
            }
        
    Returns:
        Dict containing generated question, potential mistakes, reasons and approaches
    """
    try:
        print(f"Generating personalized question for {student_data.get('Name', 'student')} in grade {student_data.get('Grade', 'unknown')}...")
        
        # Create a prompt based on the student info
        prompt = f"""Based on the following student information, generate ONE personalized math question:

Student Name: {student_data.get('Name', '')}
Age: {student_data.get('Age', '')}
Grade: {student_data.get('Grade', '')}
Subject: {student_data.get('Subject', 'Mathematics')}
Previous questions given: {student_data.get('Given-questions', 0)}
Previous correct answers: {student_data.get('Correct-answered', 0)}
Known disability: {student_data.get('Known-disability', False)}
Previously given question: {student_data.get('Given-question', '')}
Previous mistakes: {student_data.get('Mistake-made', '')}
Time taken previously: {student_data.get('Time-taken', '')}
Additional observations: {student_data.get('Additional-observation', '')}

Generate a response in JSON format with the following structure:
{{
  "Question": "The complete question text",
  "Mistakes": ["potential mistake 1", "potential mistake 2"],
  "Reasons": ["reason for potential mistake 1", "reason for potential mistake 2"],
  "Approaches": ["suggested approach 1", "suggested approach 2"]
}}
"""

        print("Sending request to OpenAI...")
        
        # Call OpenAI API with the new client format
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can use gpt-4 if available for better results
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert math educator who creates personalized questions for students based on their profile, learning history, and specific needs."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )

        print("Response received from OpenAI.")
        
        # Parse the response
        content = json.loads(response.choices[0].message.content)
        
        return content
    except Exception as error:
        print(f"Error in OpenAI question generation: {str(error)}")
        
        # More detailed error logging
        if hasattr(error, "response"):
            print(f"OpenAI API Error Response: {error.response}")
        
        raise Exception(f"Failed to generate question with OpenAI: {str(error)}")

# Simple test function
async def test_openai_connection():
    """Simple function to test OpenAI connection"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ]
        )
        return {"success": True, "message": response.choices[0].message.content}
    except Exception as e:
        print(f"OpenAI connection test failed: {str(e)}")
        return {"success": False, "error": str(e)}