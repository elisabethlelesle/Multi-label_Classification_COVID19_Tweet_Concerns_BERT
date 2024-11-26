import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
import torch.nn.functional as F
import optuna
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

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

label_thresholds = [0.5] * len(label_to_index)  # Initial thresholds for each label

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma, class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.class_weights is not None:
            focal_loss *= self.class_weights
        return focal_loss.mean()

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

def calculate_class_weights(data):
    labels = []
    for item in data:
        label_vector = [0] * len(label_to_index)
        for category in item.get('labels', {}):
            label_vector[label_to_index[category]] = 1
        labels.append(label_vector)
    labels = np.array(labels)
    class_weights = []
    for i in range(len(label_to_index)):
        weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=labels[:, i])
        class_weights.append(weights[1])
    return torch.tensor(class_weights)

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    model = model.train()
    losses = []
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits, labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(losses)

def predict_with_label_thresholds(model, data_loader, device, label_thresholds):
    model = model.eval()
    predictions = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            binary_preds = np.array([(probs[:, i] >= label_thresholds[i]).astype(int) for i in range(len(label_thresholds))]).T
            predictions.append(binary_preds)
    return np.vstack(predictions)

def objective(trial):
    model_name = trial.suggest_categorical('model_name', ['roberta-large', 'distilroberta-base', 'albert-base-v2'])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 5e-5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.4)
    alpha = trial.suggest_uniform('alpha', 0.5, 2.0)
    gamma = trial.suggest_uniform('gamma', 1, 3)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_index))
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    class_weights = calculate_class_weights(train_data).to(device)
    loss_fn = FocalLoss(alpha=alpha, gamma=gamma, class_weights=class_weights)

    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for epoch in range(epochs):
        train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
    
    # Validation score as objective for Optuna
    val_predictions = predict_with_label_thresholds(model, val_loader, device, label_thresholds)
    val_f1 = f1_score(val_targets, val_predictions, average='macro')
    return val_f1

if __name__ == "__main__":
    epochs = 5

    # Load dataset
    with open('train.json') as f:
        train_data = json.load(f)
    with open('val.json') as f:
        val_data = json.load(f)
    with open('test.json') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    train_dataset = TweetDataset(train_data, tokenizer, max_len=128)
    val_dataset = TweetDataset(val_data, tokenizer, max_len=128)
    test_dataset = TweetDataset(test_data, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    val_targets = []
    for item in val_data:
        target = [0] * len(label_to_index)
        for category in item.get('labels', {}):
            target[label_to_index[category]] = 1
        val_targets.append(target)
    val_targets = np.array(val_targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameter Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best_trial = study.best_trial

    print(f"Best trial parameters: {best_trial.params}")

    # Load best model for ensemble and final predictions
    best_model_names = ['roberta-large', 'distilroberta-base', 'albert-base-v2']
    models = []
    for model_name in best_model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_index))
        model.config.hidden_dropout_prob = best_trial.params['dropout']
        model.config.attention_probs_dropout_prob = best_trial.params['dropout']
        model = model.to(device)
        models.append(model)

    # Ensemble Predictions and Generate submission.csv
    predictions = np.mean([predict_with_label_thresholds(model, test_loader, device, label_thresholds) for model in models], axis=0)
    submission_df = pd.DataFrame(predictions.round().astype(int), columns=list(label_to_index.keys()))
    submission_df.insert(0, 'index', range(len(submission_df)))
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved as submission.csv")
