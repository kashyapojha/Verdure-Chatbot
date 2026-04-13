from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/chat")
def chat(query: Query):
    response = predict(query.text)
    return {"response": response}