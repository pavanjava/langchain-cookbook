from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from src.util import setEnv

setEnv()

template = """Question: {question}
Answer:  """

wikiSQL = 'mrm8488/t5-base-finetuned-wikiSQL'
flan_t5 = 'google/flan-t5-xxl'
databricks_dolly = 'databricks/dolly-v2-3b'

prompt = PromptTemplate(template=template, input_variables=['question'])
llm = HuggingFaceHub(repo_id=databricks_dolly, model_kwargs={'temperature': 0.5, 'max_length': 128})

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What are transformers in NLP ?"

print(llm_chain.run(question))
