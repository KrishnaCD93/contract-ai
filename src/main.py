# backend/src/main.py
import asyncio
import json
import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse

load_dotenv(find_dotenv())

TEMPLATE = """You are ProjectGPT, an AI agent with a mission to help your users understand their projects and help them achieve their goals. 
You are a friendly AI whose mission is to create robust dashboards that help users achive the mission set in their contract. 
Your mission is to always help the client get better and more comprehensive insights into their problems so they can satisfy their clients.
If the project's mission or goal is not clear from the initial contract, you are to first clarify these points so that you can complete your mission of creating robust dashboards. 
You are an agent that helps people understand, write, and manage a robust contract. 
The knowledge you have is based on your vast understaing of contract law the experience you are to emulate is that of an experienced consultant with many years of experience as a data analyst.
For any and all responses you provide, you think of a multitude of different outputs and then combine them into the best designed, most comprehensive, most accurate graphs, which you output. 
Always remember that the end goal is to create a data analytics dashboard that helps the client achieve their goals, and all your responses should be geared towards that end.
"""


class Message(BaseModel):
    human_message: Optional[str] = None
    ai_message: Optional[str] = None


app = FastAPI()

origins = [
    "https://contractcanvas.vercel.app",
    "wss://contractcanvas.vercel.app",
    "http://localhost",
    "http://localhost:3000",
    "ws://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/embed")
async def embed(input: str):
    embeddings_agent = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = embeddings_agent.embed_query(input)
    if len(embeddings) > 0:
        return {"embedding": embeddings, "error": None}
    else:
        return {"embedding": None, "error": "Error retrieving embeddings"}


# @input: messages: dictionary of messages [{ aiMessage: "message", humanMessage: "message" }]
# @message: a list of strings starting with template then alternating between ai and human messages
# @output: AI response: string
@app.post("/chat")
async def chat_sse(chat: List[Message]):
    chat_model = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        verbose=True,
        temperature=0.7,
        model_name="gpt-3.5-turbo",
    )

    async def event_stream():
        last_index = len(chat) - 1
        while True:
            if last_index < len(chat):
                try:
                    prompt = []
                    for message in chat:
                        if message.human_message:
                            prompt.append(HumanMessage(content=message.human_message))
                        if message.ai_message:
                            prompt.append(AIMessage(content=message.ai_message))
                    messages = [SystemMessage(content=TEMPLATE), *prompt]
                    ai_response = chat_model(messages).content
                    yield f"data: {ai_response}\n\n"
                    last_index = len(chat)
                except Exception as e:
                    yield f"data: Error: {str(e)}\n\n"
            await asyncio.sleep(1)

    return EventSourceResponse(event_stream())
