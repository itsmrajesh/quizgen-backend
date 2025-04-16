from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class QuestionNAnswer(BaseModel):
    question: str = Field(..., description="One of the questions from the given topic")
    options: List[str] = Field(..., description="List of possible answers")
    correct_answer: str = Field(..., description="The correct answer from the options")

class TestPaper(BaseModel):
    title: str = Field(..., description="Title of the test")
    questions: List[QuestionNAnswer] = Field(..., description="List of questions in the test")

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