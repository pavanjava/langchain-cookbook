from dotenv import load_dotenv, find_dotenv
import os


def setEnv():
    # read the local env file to get the open_api_key
    _ = load_dotenv(find_dotenv())
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
    os.environ['SERPAPI_API_KEY'] = os.getenv('SERP_API_KEY')
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACE_API_KEY')
