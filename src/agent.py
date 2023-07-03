import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

load_dotenv(find_dotenv())

class Agent:
    def __init__(self, model_name, temperature):
        self.llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name, temperature=temperature)

    def get_response(self, prompt):
        return self.llm.predict(prompt)
    
def get_embedding(input):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return embeddings.embed_query(input)