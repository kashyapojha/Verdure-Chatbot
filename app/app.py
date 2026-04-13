from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle
import boto3
import os

app = FastAPI()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Globals
model = None
tokenizer = None
label_encoder = None

# S3 Config
BUCKET = "verdure-models"   # change if needed
REGION = "ap-south-1"

s3 = boto3.client("s3", region_name=REGION)


# 🔽 Download model from S3
def download_model():
    print("Downloading model from S3...")

    structure = {
        "bert_model": ["config.json", "model.safetensors"],
        "bert_tokenizer": ["tokenizer.json", "tokenizer_config.json"],
        "": ["label_encoder.pkl"]   # root file
    }

    for folder, files in structure.items():
        for file in files:
            local_path = f"models/{folder}/{file}" if folder else file
            s3_path = f"{folder}/{file}" if folder else file

            # ✅ FIX: only create dir if exists
            dir_name = os.path.dirname(local_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # download only if not present
            if not os.path.exists(local_path):
                print(f"Downloading {s3_path}...")
                s3.download_file(BUCKET, s3_path, local_path)

    print("Download complete ✅")


# 🔽 Startup event
@app.on_event("startup")
def startup():
    global model, tokenizer, label_encoder

    download_model()

    print("Loading model...")

    model = BertForSequenceClassification.from_pretrained("models/bert_model")
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("models/bert_tokenizer")

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    print("Model loaded successfully 🚀")


# Request schema
class Query(BaseModel):
    text: str


# Prediction function
def predict(query):
    encoding = tokenizer(
        query,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1)

    label = label_encoder.inverse_transform([pred.item()])[0]
    return label


# API endpoint
@app.post("/predict")
def get_prediction(query: Query):
    if not query.text.strip():
        return {"error": "Empty query not allowed"}

    label = predict(query.text)
    return {"prediction": label}