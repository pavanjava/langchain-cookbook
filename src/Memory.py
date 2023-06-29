# Memory
# Memory is the concept of storing and retrieving data in the process of a conversation. There are two main methods:

# 1. Based on input, fetch any relevant pieces of data
# 2. Based on the input and output, update state accordingly
# There are two main types of memory: short term and long term.
# Short term memory generally refers to how to pass data in the context of a singular conversation
# (generally is previous ChatMessages or summaries of them).
# Long term memory deals with how to fetch and update information between conversations.

from util import setEnv
from langchain import OpenAI, ConversationChain

setEnv()

llm = OpenAI()

conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input='Hello There !')
print(output)

output = conversation.predict(input="My Name is pavan and I am an AI enthsiast")
print(output)

output = conversation.predict(input="Tell me about Large Language Models ?")
print(output)



