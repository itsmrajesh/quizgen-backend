from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests
from os import environ

security = HTTPBearer()

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Google OAuth2 client ID
CLIENT_ID = environ["GOOGLE_AUTH_CLIENT_ID"]

def verify_google_token(token):
    try:
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)
        user_id = idinfo["sub"]
        email = idinfo["email"]
        name = idinfo.get("name", "service-account")
        azp = idinfo.get("azp")
        return {"user_id": user_id, "email": email, "name": name, "azp": azp}
    except ValueError as ve:
        print(f"Invalid token {str(ve)}")
        raise ValueError(f"Invalid token {str(ve)}")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    user_info = verify_google_token(token)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    return user_info


# LangChain model setup
model_name = "gpt-4o-2024-08-06"
model = ChatOpenAI(model_name=model_name, temperature=0.7)

# Output structure models
class QuestionNAnswer(BaseModel):
    question: str = Field(..., description="One of the questions from the given topic")
    options: List[str] = Field(..., description="List of possible answers")
    correct_answer: str = Field(..., description="The correct answer from the options")

class TestPaper(BaseModel):
    title: str = Field(..., description="Title of the test")
    questions: List[QuestionNAnswer] = Field(..., description="List of questions in the test")

structured_model = model.with_structured_output(TestPaper)

# Prompt template
prompt_template = ChatPromptTemplate.from_template(
    "Generate a test paper on the topic of '{topic}'. "
    "The test should have a title and at least {question_count} questions with multiple-choice answers "
    "with difficulty level {level}. Each question should have 4 options and one correct answer. "
    "The output should be in JSON format."
)

# Request and response models
class QuizRequest(BaseModel):
    topic: str
    question_count: Optional[int] = Field(10, gt=0, le=10, description="Number of questions in the test")
    level: Literal["easy", "medium", "hard"] = Field("medium", description="Difficulty level of the questions")

class QuizResponse(BaseModel):
    test_paper: TestPaper
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float

@app.post("/quiz/create", response_model=QuizResponse)
def create_quiz(request: QuizRequest, user_info = Depends(verify_token)):
    try:
        prompt = prompt_template.format_prompt(
            topic=request.topic,
            question_count=request.question_count,
            level=request.level
        )
        
        print(f"User Info: {user_info}")

        with get_openai_callback() as cb:
            response = structured_model.invoke(prompt)
            print(f"Input Tokens: {cb.prompt_tokens}")
            print(f"Output Tokens: {cb.completion_tokens}")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Cost: ${cb.total_cost:.6f}")

        test_paper = TestPaper.model_validate(response)

        return QuizResponse(
            test_paper=test_paper,
            input_tokens=cb.prompt_tokens,
            output_tokens=cb.completion_tokens,
            total_tokens=cb.total_tokens,
            cost=cb.total_cost
        )
    
    except Exception as e:
        error_message = f"Internal Server Error: {str(e)}"
        print(f"Error: {error_message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error, please try again later."
        )
