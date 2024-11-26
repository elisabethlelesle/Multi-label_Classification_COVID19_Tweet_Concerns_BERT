import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np

label_to_index = {
    "ineffective": 0,
    "unnecessary": 1,
    "pharma": 2,
    "rushed": 3,
    "side-effect": 4,
    "mandatory": 5,
    "country": 6,
    "ingredients": 7,
    "political": 8,
    "none": 9,
    "conspiracy": 10,
    "religious": 11
}

class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tweet = self.data[index]['tweet']
        inputs = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if 'labels' in self.data[index]:
            labels = self.data[index]['labels']
            label_tensor = torch.zeros(len(label_to_index))
            for category, items in labels.items():
                if category in label_to_index:
                    label_index = label_to_index[category]
                    label_tensor[label_index] = 1
            return {
                'input_ids': inputs['input_ids'].flatten(),
                'attention_mask': inputs['attention_mask'].flatten(),
                'labels': label_tensor
            }
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten()
        }

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return np.mean(losses)

def predict(model, data_loader, device, threshold=0.5):
    model = model.eval()
    predictions = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()  # Convert to probabilities
            binary_preds = (probs >= threshold).astype(int)  # Apply threshold
            predictions.append(binary_preds)
    return np.vstack(predictions)

if __name__ == "__main__":
    # Load dataset
    with open('train.json') as f:
        train_data = json.load(f)
    with open('val.json') as f:
        val_data = json.load(f)
    with open('test.json') as f:
        test_data = json.load(f)
        
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TweetDataset(train_data, tokenizer, max_len=128)
    val_dataset = TweetDataset(val_data, tokenizer, max_len=128)
    test_dataset = TweetDataset(test_data, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_index))
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 10

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, torch.nn.BCEWithLogitsLoss(), optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train loss: {train_loss}")
    
    # Predict, threshold, and save submission
    predictions = predict(model, test_loader, device)
    submission_df = pd.DataFrame(predictions, columns=list(label_to_index.keys()))
    submission_df.insert(0, 'index', range(len(submission_df)))
    submission_df.to_csv("submission.csv", index=False)

