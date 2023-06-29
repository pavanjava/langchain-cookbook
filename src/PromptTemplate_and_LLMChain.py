# LLMChain:
# A LLMChain is the most common type of chain. It consists of a PromptTemplate, a model (either an LLM or a ChatModel),
# and an optional output parser. This chain takes multiple input variables,
# uses the PromptTemplate to format them into a prompt. It then passes that to the model.
# Finally, it uses the OutputParser (if provided) to parse the output of the LLM into a final format.

# Prompt Template
# A PromptValue is what is eventually passed to the model. Most of the time,
# this value is not hardcoded but is rather dynamically created based on a combination of user input,
# other non-static information (often coming from multiple sources), and a fixed template string.
# We call the object responsible for creating the PromptValue a PromptTemplate.
# This object exposes a method for taking in input variables and returning a PromptValue.

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from util import setEnv

setEnv()

prompt = PromptTemplate(
    input_variables=['language'],
    template='explain me about {language} ?'
)

llm = OpenAI(temperature=0.6, model_name="text-davinci-003")

llm_chain = LLMChain(llm=llm, prompt=prompt)

langauge = 'Redux'
print('Your Prompt is: '+prompt.format(language=langauge))
print(llm_chain.run(langauge))


