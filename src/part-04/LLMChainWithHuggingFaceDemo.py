from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from src.util import setEnv

setEnv()

template = """Question: {question} expand the answer in about 20 sentences
Answer: """

prompt = PromptTemplate(template=template, input_variables=['question'])
llm = HuggingFaceHub(repo_id='google/flan-t5-xxl', model_kwargs={'temperature': 0.5, 'max_length': 128})

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is Area 51 ?"

print(llm_chain.run(question))
