from fastapi import FastAPI
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained("models/bert_model")
tokenizer = BertTokenizer.from_pretrained("models/bert_tokenizer")

model.to(device)
model.eval()

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict(query):
    encoding = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(output.logits, dim=1).item()

    return label_encoder.inverse_transform([pred])[0]

@app.get("/predict")
def get_prediction(q: str):
    return {"query": q, "prediction": predict(q)}