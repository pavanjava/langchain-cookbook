# Get the OpenAI llm from langchain
from langchain.llms import OpenAI
from util import setEnv

setEnv()

# instantiate the LLM
llm = OpenAI()

# now call the LLM from code
text = 'What is the scientific name of vitamin B4'
print(llm(text))