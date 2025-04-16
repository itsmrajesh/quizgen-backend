from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from models import QuizRequest, TestPaper, QuizResponse
from langchain_community.callbacks.manager import get_openai_callback
from os import getenv

model_name = getenv("LLM_MODEL_ID", "gpt-4o-2024-08-06")
temprature = float(getenv("LLM_TEMPERATURE", 0.7))

model = ChatOpenAI(model_name=model_name, temperature=0.7)

structured_model = model.with_structured_output(TestPaper)

prompt_template = ChatPromptTemplate.from_template(
    "Generate a test paper on the topic of '{topic}'. "
    "The test should have a title and at least {question_count} questions with multiple-choice answers "
    "with difficulty level {level}. Each question should have 4 options and one correct answer. "
    "The output should be in JSON format."
)


def invoke_llm(request: QuizRequest):
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