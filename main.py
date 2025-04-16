from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests
from os import environ
from models import QuizRequest, QuizResponse
from llm import invoke_llm
from sqlalchemy.orm import Session
from db import get_db, QuizCreate, insert_quiz, check_user_cost_limit

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


@app.post("/quiz/create", response_model=QuizResponse)
def create_quiz(request: QuizRequest, user_info = Depends(verify_token), db: Session = Depends(get_db)):
    try:

        cost = check_user_cost_limit(user_info["email"], db)

        if cost > 1:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your Quiz generation limit exceeded."
            )

        quiz_response = invoke_llm(request)

        db_quiz = QuizCreate(
            name=user_info["name"],
            email=user_info["email"],
            quiz_title=request.topic,
            difficulty_level=request.level,
            no_of_questions=request.question_count,
            cost=quiz_response.cost
        )

        insert_quiz(db_quiz, db)

        return quiz_response

    except HTTPException as http_exc:
        print(f"HTTP Exception: {http_exc.detail}")
        return JSONResponse(
            status_code=http_exc.status_code,
            content={"detail": http_exc.detail}
        )
    
    except Exception as e:
        error_message = f"Internal Server Error: {str(e)}"
        print(f"Error: {error_message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": error_message}
        )
