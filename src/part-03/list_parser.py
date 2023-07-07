from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from typing import Any
from src.util import setEnv

setEnv()
output_parser = CommaSeparatedListOutputParser()


def execute():
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="List five {programing_languages_or_frameworks}.\n{format_instructions}",
        input_variables=["programing_languages_or_frameworks"],
        partial_variables={"format_instructions": format_instructions}
    )
    model = OpenAI(temperature=0)
    _input = prompt.format(programing_languages_or_frameworks="web frameworks in javascript")
    output = model(_input)
    format_output(output=output)


def format_output(output: Any):
    print(output_parser.parse(output))


if __name__ == "__main__":
    execute()
