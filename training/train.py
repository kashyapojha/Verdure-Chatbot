import pandas as pd
import numpy as np
import torch
import pickle
import mlflow
import mlflow.pytorch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset


# ======================
# LOAD DATA
# ======================
data = pd.read_csv('final_data.csv')

label_encoder = LabelEncoder()
data['id'] = label_encoder.fit_transform(data['id'])

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# ======================
# TOKENIZER
# ======================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class QueryDataset(Dataset):
    def __init__(self, queries, labels, tokenizer, max_len):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = str(self.queries[index])
        label = self.labels[index]

        encoding = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = QueryDataset(
    train_data['user_query'].values,
    train_data['id'].values,
    tokenizer,
    128
)

test_dataset = QueryDataset(
    test_data['user_query'].values,
    test_data['id'].values,
    tokenizer,
    128
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


# ======================
# MODEL
# ======================
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(data['id'].unique())
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


# ======================
# TRAIN FUNCTION (FIXED)
# ======================
def train_epoch(model, loader, optimizer):
    model.train()

    losses = []
    correct = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        losses.append(loss.item())

        loss.backward()

        # ✅ IMPORTANT: prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

    acc = correct / len(loader.dataset)
    return acc, np.mean(losses)


# ======================
# EVAL FUNCTION (FIXED)
# ======================
def eval_model(model, loader):
    model.eval()

    losses = []
    correct = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            losses.append(loss.item())

    acc = correct / len(loader.dataset)
    return acc, np.mean(losses)


# ======================
# MLFLOW TRAINING
# ======================
mlflow.set_experiment("verdure-chatbot")

EPOCHS = 3

with mlflow.start_run():

    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("lr", 2e-5)
    mlflow.log_param("model", "bert-base-uncased")
    mlflow.log_param("batch_size", 16)

    for epoch in range(EPOCHS):

        train_acc, train_loss = train_epoch(model, train_loader, optimizer)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)

    test_acc, test_loss = eval_model(model, test_loader)

    print(f"\nTest Accuracy: {test_acc:.4f}")

    mlflow.log_metric("test_acc", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    # save model to MLflow
    mlflow.pytorch.log_model(model, "model")


# ======================
# SAVE ARTIFACTS LOCALLY
# ======================
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_tokenizer")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nTraining + MLflow tracking completed successfully!")