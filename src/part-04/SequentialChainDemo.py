from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from src.util import setEnv

setEnv()

llm = OpenAI(temperature=0.8)

translation_template = """ you are the content writer. translate the given article into hindi 
    article: {article}
    translation: this is the translation of the above article:
"""
translation_prompt_template = PromptTemplate(input_variables=['article'], template=translation_template)
# translation_chain = LLMChain(llm=llm, prompt=translation_prompt_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt_template, output_key='translated_article')

summarization_template = """you are the content writer. summarize the given translation into 10 words
{translated_article}
summary of the article:
"""
summarization_prompt_template = PromptTemplate(input_variables=['translated_article'], template=summarization_template)
# summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt_template, output_key='summarized')

# overall_chain = SimpleSequentialChain(chains=[translation_chain, summarization_chain], verbose=True)
overall_chain = SequentialChain(chains=[translation_chain, summarization_chain],
                                input_variables=['article'],
                                output_variables=['translated_article', 'summarized'],
                                verbose=True)

# print(overall_chain.run('The next step after calling a language model is make a series of calls to a '
#                         'language model. This is particularly useful when you want to take the output '
#                         'from one call and use it as the input to another.'))

print(overall_chain({'article': 'The next step after calling a language model is make a series of calls to a '
                                'language model. This is particularly useful when you want to take the output '
                                'from one call and use it as the input to another.'}))
