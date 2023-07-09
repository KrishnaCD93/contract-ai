import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv(find_dotenv())

TEMPLATE = """You are contractGPT, a chatbot who has learned and amalgamated knowledge from many expert contracts and contract lawyers.
    You are here to help people understand their contracts and the law. 
    You also have expert knowledge of the law and can draft contracts with all legalese included."""
    
class ChatBot:
    def __init__(self, model_name, temperature):
        self.chat_model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=temperature, model_name=model_name)
    
    def get_response(self, prompt):
        messages = [SystemMessage(content=TEMPLATE), HumanMessage(content=prompt)]
        return self.chat_model(messages)
    
def get_embedding(input):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return embeddings.embed_query(input)