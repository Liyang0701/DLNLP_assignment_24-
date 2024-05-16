import numpy as np 
import pandas as pd 
from ast import literal_eval
from itertools import chain
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import torch
from transformers import AutoTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

# Path to dataset
USER= "./Datasets"

# Function to create a dataframe
def create_df(debug=False):
    # Read CSV files
    feats = pd.read_csv(f"{USER}/features.csv")
    notes = pd.read_csv(f"{USER}/patient_notes.csv")
    train = pd.read_csv(f"{USER}/train.csv")
    
    # Convert string representations of lists to actual lists
    train["annotation_list"] = train["annotation"].apply(literal_eval)
    train["location_list"] = train["location"].apply(literal_eval)
    
    # Merge DataFrames
    df = train.merge(notes, how="left").merge(feats, how="left")
    
    # Filter out rows where 'annotation' is an empty list
    df = df[df["annotation"] != "[]"].copy().reset_index(drop=True)
    
    # Clean and preprocess text columns
    df["feature_text"] = df["feature_text"].str.replace("-OR-", ";-").str.replace("-", " ").str.lower()
    df["pn_history"] = df["pn_history"].str.lower()
    
    # If debug is True, sample 50% of the data
    if debug:
        df = df.sample(frac=0.5).reset_index(drop=True)
    
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5)
    df["stratify_on"] = df["case_num"].astype(str) + df["feature_num"].astype(str)
    df["fold"] = -1
    for fold, (_, valid_idx) in enumerate(skf.split(df["id"], df["stratify_on"])):
        df.loc[valid_idx, "fold"] = fold
    
    return df

# Determine maximum token length
def determine_max_len():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    feats = pd.read_csv(f"{USER}/features.csv")
    notes = pd.read_csv(f"{USER}/patient_notes.csv")
    
    # Get token lengths for features and patient notes
    feature_token_lengths = feats['feature_text'].apply(lambda x: len(tokenizer.tokenize(x)))
    pn_history_token_lengths = notes['pn_history'].apply(lambda x: len(tokenizer.tokenize(x)))
    
    # Plot histogram of token lengths
    plt.figure(figsize=(12, 6))
    plt.hist(feature_token_lengths, bins=50, alpha=0.5, label='feature_text')
    plt.hist(pn_history_token_lengths, bins=50, alpha=0.5, label='pn_history')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title('Distribution of Token Lengths')
    plt.legend()
    plt.show()

# Convert location lists to integer pairs
def loc_list_to_ints(loc_list):
    to_return = []
    for loc_str in loc_list:
        loc_strs = loc_str.split(";")
        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))
    return to_return

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and label the input data
def tokenize_and_label(tokenizer, example):
    tokenized_inputs = tokenizer(
        example["feature_text"],
        example["pn_history"],
        truncation = "only_second",
        max_length = 406, 
        padding = "max_length",
        return_offsets_mapping = True
    )
    labels = [0.0] * len(tokenized_inputs["input_ids"])
    tokenized_inputs["location_int"] = loc_list_to_ints(example["location_list"])
    tokenized_inputs["sequence_ids"] = tokenized_inputs.sequence_ids()

    for idx, (seq_id, offsets) in enumerate(zip(tokenized_inputs["sequence_ids"], tokenized_inputs["offset_mapping"])):
        if seq_id is None or seq_id == 0:
            labels[idx] = -100
            continue
        exit = False
        token_start, token_end = offsets
        for feature_start, feature_end in tokenized_inputs["location_int"]:
            if exit:
                break
            if token_start >= feature_start and token_end <= feature_end:
                labels[idx] = 1.0
                exit = True
    tokenized_inputs["labels"] = labels
    
    return tokenized_inputs

# Dataset class for handling data
class Data(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data.loc[idx]
        tokenized = tokenize_and_label(self.tokenizer, example)
        input_ids = np.array(tokenized["input_ids"]) # for input BERT
        attention_mask = np.array(tokenized["attention_mask"]) # for input BERT
        labels = np.array(tokenized["labels"]) # for calculate loss and cv score
        offset_mapping = np.array(tokenized["offset_mapping"]) # for calculate cv score
        sequence_ids = np.array(tokenized["sequence_ids"]).astype("float16") # for calculate cv score
        
        return input_ids, attention_mask, labels, offset_mapping, sequence_ids
    
# Model class for BERT
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(p = 0.2)
        self.classifier = torch.nn.Linear(768, 1) # BERT has last_hidden_state(size: sequence_length, 768)
    
    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.backbone(input_ids = input_ids, attention_mask = attention_mask)[0] 
        logits = self.classifier(self.dropout(last_hidden_state)).squeeze(-1)
        return logits

# Hyperparameters and device setup
val_fold = 0
test_fold = 1
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Model().to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)

# Create train, validation, and test datasets and dataloaders
def create_sets(df):
    train = df.loc[~df["fold"].isin([test_fold, val_fold])].reset_index(drop = True)
    valid = df.loc[df["fold"] == val_fold].reset_index(drop = True)
    test = df.loc[df["fold"] == test_fold].reset_index(drop = True)
    test.to_csv("test.csv", index=False) # save test dataset to test.csv
    train_ds = Data(train, tokenizer)
    valid_ds = Data(valid, tokenizer)
    test_ds = Data(test, tokenizer)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size = BATCH_SIZE, pin_memory = True, shuffle = True, drop_last = True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size = BATCH_SIZE * 2, pin_memory = True, shuffle = False, drop_last = False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size = BATCH_SIZE * 2, pin_memory = True, shuffle = False, drop_last = False)

    return train_dl, valid_dl, test_dl

# Utility class to keep track of averages
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Sigmoid function for predictions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Get location predictions from model outputs
def get_location_predictions(preds, offset_mapping, sequence_ids, test=False):
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        pred = sigmoid(pred)
        start_idx, current_preds = None, []
        for p, o, s_id in zip(pred, offsets, seq_ids):
            if s_id is None or s_id == 0:
                continue
            if p > 0.5:
                if start_idx is None:
                    start_idx = o[0]
                end_idx = o[1]
            elif start_idx is not None:
                current_preds.append(f"{start_idx} {end_idx}" if test else (start_idx, end_idx))
                start_idx = None
        all_predictions.append("; ".join(current_preds) if test else current_preds)
    return all_predictions

# Calculate metrics for predictions
def calculate_metrics(predictions, offset_mapping, sequence_ids, labels):
    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels in zip(predictions, offset_mapping, sequence_ids, labels):
        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros((num_chars))
        for o, s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0:
                continue
            if int(label) == 1:
                char_labels[o[0]:o[1]] = 1
        char_preds = np.zeros((num_chars))
        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1
        all_labels.extend(char_labels)
        all_preds.extend(char_preds)
    results = precision_recall_fscore_support(all_labels, all_preds, average = "binary")
    accuracy = accuracy_score(all_labels, all_preds)
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2],
        "accuracy": accuracy
    }

# Plot training and validation loss
def loss_plot(history):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), history["train"], label="Training Loss")
    plt.plot(range(1, EPOCHS + 1), history["valid"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

