from typing import Any
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage, HumanMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
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
        SystemMessage(content='Assume you are a helpful assistant that translates English to Hindi.'),
        HumanMessage(content='I love programming.')
    ]
    generate(llm=llm, text=messages)

    # making use of prompt templates
    template = 'You are a helpful assistant that completes the given {input_sentence}.'
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = '{text}'
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    text = chat_prompt.format_prompt(input_sentence='context in triple tick as shown in ```', text='Explain me something '
                                                                                                   'about java in the '
                                                                                                   'form of '
                                                                                                   '``` '
                                                                                                   '1. the first point \
                                                                                                    2. the second point \
                                                                                                    3. the third point ```'
                                                                                                   'so on.').to_messages()
    generate(llm=llm, text=text)
