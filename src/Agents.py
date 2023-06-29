# Agents:
# Some applications will require not just a predetermined chain of calls to LLMs/other tools,
# but potentially an unknown chain that depends on the user's input.
# In these types of chains, there is a “agent” which has access to a suite of tools. Depending on the user input,
# the agent can then decide which, if any, of these tools to call.

# Tools: How language models interact with other resources.
# Agents: The language model that drives decision making.
# Toolkits: Sets of tools that when used together can accomplish a specific task.
# Agent Executor: The logic for running agents with tools.

from util import setEnv

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI


setEnv()

# let's load the language model
llm = OpenAI(temperature=0.7, model_name="text-davinci-003")

# secondly load some tools to utilise like llm-math, serpapi etc
tools = load_tools(['serpapi'])

# lastly initialise the agent with tools, LLM & agent
agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Trigger the agent and observer the console
agent.run('What are the latest features of NextJS v12.1 ?')
