from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv()

model_name = "gpt-4o-2024-08-06"

model = ChatOpenAI(model_name=model_name, temperature=0.7)


class QuestionNAnswer(BaseModel):
    question: str = Field(..., description="One the question from the given topic")
    options: List[str] = Field(..., description="List of possible answers")
    correct_answer: str = Field(..., description="The correct answer from the options")


class TestPaper(BaseModel):
    title: str = Field(..., description="Title of the test")
    questions: List[QuestionNAnswer] = Field(..., description="List of questions in the test")



structured_model  = model.with_structured_output(TestPaper)


prompt_template = ChatPromptTemplate.from_template(
    "Generate a test paper on the topic of '{topic}'. "
    "The test should have a title and at least {question_count} questions with multiple-choice answers "
    "with difficulty level {level}. Each question should have 4 options and one correct answer. "
    "The output should be in JSON format."
)

prompt = prompt_template.format_prompt(
    topic="Python programming",
    question_count=5,
    level="hard"
)


with get_openai_callback() as cb:
    response = structured_model.invoke(prompt)
    print(cb)
    input_token = cb.prompt_tokens
    output_token = cb.completion_tokens
    total_token = cb.total_tokens
    cost = cb.total_cost
    print(f"Input Tokens: {input_token}")
    print(f"Output Tokens: {output_token}")
    print(f"Total Tokens: {total_token}")
    print(f"Cost: {cost}")

# Validate the response
test_paper = TestPaper.model_validate(response)
print(test_paper)