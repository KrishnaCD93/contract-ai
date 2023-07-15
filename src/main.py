from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.agent import Chatbot
from src.agent import get_embedding
import json

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
    predition = agent.get_response(inputs=messages)
    if predition:
        return {"prediction": predition, "error": None}
    else:
        return {"prediction": None, "error": "Error retrieving prediction"}

@app.get("/embed/{input}")
async def embed(input: str):
    embedding = get_embedding(input)
    if embedding.len > 0:
        return {"embedding": embedding, "error": None}
    else:
        return {"embedding": None, "error": "Error retrieving embeddings"}