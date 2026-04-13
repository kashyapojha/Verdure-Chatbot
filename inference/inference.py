import mlflow.pytorch
import torch
from transformers import BertTokenizer
import pickle

# ======================
# LOAD MLflow MODEL
# ======================
MODEL_URI = "runs:/<RUN_ID>/model"  
# (we will fix RUN_ID below)

model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ======================
# LOAD TOKENIZER
# ======================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ======================
# LOAD LABEL ENCODER
# ======================
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# ======================
# PREDICTION FUNCTION
# ======================
def predict(query):
    encoding = tokenizer(
        query,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

    label_id = preds.item()
    return label_encoder.inverse_transform([label_id])[0]