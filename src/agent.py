# backend/src/agent.py

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

TEMPLATE = '''You are contractGPT. An AI agent with a mission to help your users understand and write new contracts that help them achieve their goals. You are a friendly AI that is here to help. You are not a lawyer and you are not giving legal advice. 
You are a tool that helps people understand, write, and manage a robust contract. The knowledge you have is based on your vast understaing of contract law the experience you are to emulate is that of an experienced contract lawyer with 20 years of experience at a major law firm.
Finally, for any and all responses you provide, you think 10x the answers and then combine them into the best worded, most comprehensive, most accurate answer, which you output. Your mission is to always provide better and more comprehensive answers.
'''

class Chatbot:
    def __init__(self, model_name, temperature):
        self.chat_model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=temperature, model_name=model_name)
    
    # @input: messages: dictionary of messages [{ aiMessage: "message", humanMessage: "message" }]
    # @message: a list of strings starting with template then alternating between ai and human messages
    # @output: AI response: string
    def get_response(self, inputs):
        prompt = []
        for input in inputs:
            if input["aiMessage"]:
                prompt.append(AIMessage(content=input["aiMessage"]))
            if input["humanMessage"]:
                prompt.append(HumanMessage(content=input["humanMessage"]))
        if prompt:  # check if there is at least one prompt from the user
            messages = [SystemMessage(content=TEMPLATE), *prompt]
            print(f"input message: {messages}")
            return self.chat_model(messages).content
    
def get_embedding(input):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return embeddings.embed_query(input)