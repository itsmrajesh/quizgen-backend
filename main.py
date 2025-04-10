from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv()

app = FastAPI()

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
    question_count: Optional[int] = Field(10, gt=0, le=20, description="Number of questions in the test")
    level: Literal["easy", "medium", "hard"] = Field("medium", description="Difficulty level of the questions")

class QuizResponse(BaseModel):
    test_paper: TestPaper
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float

@app.post("/quiz/create", response_model=QuizResponse)
def create_quiz(request: QuizRequest):
    prompt = prompt_template.format_prompt(
        topic=request.topic,
        question_count=request.question_count,
        level=request.level
    )

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
