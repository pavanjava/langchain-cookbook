from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from src.util import setEnv

setEnv()

model_name = 'text-davinci-003'
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to ask on any honeywell product")
    punchline: str = Field(description="answer to describe the honeywell product")

    # You can add custom validation logic easily with Pydantic.
    @validator('setup')
    def question_ends_with_question_mark(cls, field):
        if field[-1] != '?':
            raise ValueError("Badly formed question!")
        return field


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# And a query intended to prompt a language model to populate the data structure.
product_query = "Explain me about Honeywell CK65 ?"
_input = prompt.format_prompt(query=product_query)

output = model(_input.to_string())

print(f'Question: {parser.parse(output).setup}')
print(f'Answer: {parser.parse(output).punchline}')