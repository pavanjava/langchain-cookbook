from typing import Any
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage, HumanMessage
)
from src.util import setEnv


def generate(llm: OpenAI | ChatOpenAI, text: Any):
    if type(llm) is OpenAI:
        print("Using LLMs Language Model \n")
        llm_response = llm.generate(text)
        print(llm_response.generations[0][0].text)
        print(llm_response.generations[0][0].generation_info)
    elif type(llm) is ChatOpenAI:
        print("Using Chat Models from Language Model \n")
        response = llm(text)
        print(response)


if __name__ == '__main__':
    setEnv()
    llm = OpenAI(temperature=0.5)
    text = ['I Love coding AI, ML, DL & NLP']
    generate(llm=llm, text=text)

    # implementation of chat model, which take list of messages
    llm = ChatOpenAI()
    messages = [
        SystemMessage(content="Assume you are a helpful assistant that translates English to Hindi."),
        HumanMessage(content="I love programming.")
    ]
    generate(llm=llm, text=messages)
