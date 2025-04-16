from pydantic import BaseModel
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
from sqlalchemy.orm import Session

import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class QuizCreate(BaseModel):
    name: str
    email: str
    quiz_title: str
    difficulty_level: str
    no_of_questions: int
    cost: float


Base = declarative_base()


class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False)
    quiz_title = Column(String(255), nullable=False)
    difficulty_level = Column(String(50))
    no_of_questions = Column(Integer)
    cost = Column(Float, default=0.0)


def insert_quiz(quiz: QuizCreate, db_session: Session):
    try:
        db_quiz = Quiz(
            name=quiz.name,
            email=quiz.email,
            quiz_title=quiz.quiz_title,
            difficulty_level=quiz.difficulty_level,
            no_of_questions=quiz.no_of_questions,
            cost=quiz.cost,
        )
        db_session.add(db_quiz)
        db_session.commit()
        db_session.refresh(db_quiz)
        print(f"Quiz inserted with ID: {db_quiz.id}")
    except Exception as e:
        print(f"Error inserting quiz: {e}")


def check_user_cost_limit(user_email: str, db_session: Session):
    try:
        total_cost = (
            db_session.query(func.sum(Quiz.cost))
            .filter(Quiz.email == user_email)
            .scalar()
            or 0
        )
        return total_cost
    except Exception as e:
        print(f"Error checking user cost limit: {e}")
        return 0.0
