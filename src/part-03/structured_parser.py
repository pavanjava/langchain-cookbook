from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from src.util import setEnv

setEnv()

response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

model = OpenAI(temperature=0)
_input = prompt.format_prompt(question="inventor of springboot?")
output = model(_input.to_string())
print(output_parser.parse(output))