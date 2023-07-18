# backend/src/main.py

import time
from typing import List, Union, Generator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.agent import Chatbot, get_embedding
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
import json

class Message(BaseModel):
    humanMessage: str
    aiMessage: str

class Messages(BaseModel):
    messages: List[Message]

app = FastAPI()

origins = [
    "https://contractcanvas.vercel.app",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/{input}")
async def predict(input: str):
    print(f"Input: {input}")
    messages = json.loads(input)
    print(f"Messages: {messages}")
    agent = Chatbot(model_name="gpt-3.5-turbo-0613", temperature=0.9)

    def event_stream() -> Generator:
        while True:
            prediction = agent.get_response(inputs=messages)
            if prediction:
                yield f"data: {prediction}\n\n"
            else:
                # Wait for a bit before checking for new messages
                time.sleep(0.1)


    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/embed/")
async def embed(input: str):
    embedding = get_embedding(input)
    if embedding.len > 0:
        return {"embedding": embedding, "error": None}
    else:
        return {"embedding": None, "error": "Error retrieving embeddings"}