# Get the OpenAI llm from langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from util import setEnv

setEnv()

# instantiate the LLM model
llm = OpenAI()

# now call the LLM from code
text = 'What is the scientific name of vitamin B4'
print(llm(text))

# instantiate the Chat model
chat = ChatOpenAI()
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love coding.")
]
print(chat(messages=messages))
