from langchain import PromptTemplate, OpenAI, LLMChain
from src.util import setEnv

setEnv()


# compute all the differentials and integrals (tested)
def calculus_chain(computation: str, expression: str):
    prompt_template = "What is the {computation} of {expression}"
    llm = OpenAI(temperature=0.8)
    llmChain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    return llmChain.run({'computation': computation, 'expression': expression})


# can also pass {'computation': 'integral', 'expression': '1/x'}
if __name__ == '__main__':
    result = calculus_chain('differential', '1/e^x')
    print(result)
