from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.util import setEnv

setEnv()
nltk.download('stopwords')


def remove_stopwords(inputs: dict) -> dict:
    text = inputs["text"]
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return {"output_text": " ".join(filtered_sentence)}


transform_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=remove_stopwords
)

template = """Identify the sentiment of this text:

{output_text}

Summary:"""
prompt = PromptTemplate(input_variables=["output_text"], template=template)
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)
sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

response = sequential_chain.run("To tell the country’s most revered good vs evil tale, as old as the hills to the contemporary audience, "
                              "without sounding archaic is no mean task. When the content has generational awareness, novel storytelling can "
                              "be its only differentiator. Raut goes the Marvel way to draw in the younger crowd as his film rides high on "
                              "action-adventure over ethos."
                              "The narrative wastes no time in establishing characters or Ram’s aura (Prabhas as Raghav) or what led to his "
                              "exile (vanvas) from Ayodhya. It focuses on Sita’s (Kriti Sanon as Janaki) treacherous abduction by Ravan (Saif "
                              "Ali Khan) and the epic Ram vs Ravan battle fought for her rescue. The film pits Ram’s fearless army comprising "
                              "Lakshman, Hanuman, Sugriv and their vanar sena against the menacing, and towering Ravan and his immortality. "
                              "The battle scenes recreate the iconic Avengers’ huddle warding off a larger army of Ravan’s CGI rakshasas. The "
                              "war (second half) is engaging and redeems a rather stagnant first half that lacks thrill or a sense of urgency "
                              "that the story demands."
                              "Raut struggles to find a balance and consistency between the epic story and its superhero-verse execution. The "
                              "dialogue lack the impact that epic heroes of this stature are expected to deliver. Characters sound "
                              "unconvincing as they randomly oscillate between ‘Adharma ka vidhvansa’ to ‘tere baap ki jalegi and tu marega’. "
                              "The narration feels bland in the first half. It does not evoke the kind of emotional gravity that you would "
                              "expect from an epic tale like Ramayana. You don't feel invested in the characters enough."
                              "Saif Ali Khan’s invincible Ravan exudes main character energy in this ambitious but stoical retelling of an "
                              "epic. While Prabhas (voiced brilliantly by Sharad Kelkar) maintains a heroic presence as Ram, it is Saif, "
                              "with his wicked mannersims and massive height lift that steals the show. Tanhaji: The Unsung Warrior was "
                              "testament to his mastery at playing dark and delirious characters and here he raises the bar yet again. The "
                              "music and background score composed by Sanchit and Ankit Balhara, as well as the songs by Ajay-Atul give a "
                              "terrific boost to Saif’s monstrous portrayal of Ravan. Adipurush belongs to Saif Ali Khan and Raut succeeds in "
                              "mounting the character on a massive scale."
                              "The VFX and visual appeal are passable if not impressive. The 3D feels like an unnecessary accessory. With a "
                              "run time of 3 hours, you wish the story wasn’t as dependent on the special effects as it should have been on "
                              "the nature of its revered characters or what set them apart. Despite the dramatic buildup, the climax doesn't "
                              "live you with that sense of joy, reward or victory. This one’s a sincere attempt that gets a tad overwhelmed "
                              "by its ambition of handling a story of this magnitude.")

print(response)

