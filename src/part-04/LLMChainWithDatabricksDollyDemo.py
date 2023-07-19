import torch
from transformers import pipeline
from src.util import setEnv
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline


setEnv()

# download the model locally and use it
generate_text = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device_map="auto", return_full_text=True)


# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)

print(llm_chain.predict(instruction="Explain to me the difference between nuclear fission and fusion.").lstrip())