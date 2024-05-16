from A import data_preprocess
import numpy as np 
from tqdm.notebook import tqdm
import torch
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    # Path to dataset
    USER = "./Datasets"
    
    # Preprocess the data and create a dataframe
    df = data_preprocess.create_df()
    df.head()
    
    # Determine maximum token length for feature text and patient notes
    data_preprocess.determine_max_len()
    
    # Load the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Set hyperparameters and device
    val_fold = 0
    test_fold = 1
    BATCH_SIZE = 16
    EPOCHS = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the model, loss function, and optimizer
    model = data_preprocess.Model().to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)
    
    # Create dataloaders for training, validation, and test sets
    train_dl, valid_dl, test_dl = data_preprocess.create_sets(df)
    
    # Initialize history for tracking training and validation loss
    history = {"train": [], "valid": []}
    best_loss = np.inf
    
    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = data_preprocess.AverageMeter()
        pbar = tqdm(train_dl, desc=f"Training Epoch {epoch + 1}")
        
        for batch in pbar:
            input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch[:3]]
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = torch.nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
            loss = torch.masked_select(loss, labels > -1).mean()
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), n=len(input_ids))
            pbar.set_postfix(Loss=train_loss.avg)
        
        print(f"Epoch {epoch + 1} Training Loss: {train_loss.avg:.4f}")
        history["train"].append(train_loss.avg)

        # Validation phase
        model.eval()
        valid_loss = data_preprocess.AverageMeter()
        pbar = tqdm(valid_dl, desc=f"Validation Epoch {epoch + 1}")
        
        with torch.no_grad():
            for batch in pbar:
                input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch[:3]]
                
                logits = model(input_ids, attention_mask)
                loss = torch.nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
                loss = torch.masked_select(loss, labels > -1).mean()
                
                valid_loss.update(loss.item(), n=len(input_ids))
                pbar.set_postfix(Loss=valid_loss.avg)
        
        print(f"Epoch {epoch + 1} Validation Loss: {valid_loss.avg:.4f}")
        history["valid"].append(valid_loss.avg)

        # Save model if validation loss improves
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "nbme.pth")
    
    # Plot the training and validation loss
    data_preprocess.loss_plot(history)
    
    # Load the best model
    model.load_state_dict(torch.load("nbme.pth", map_location=DEVICE))

    # Evaluation on test set
    model.eval()
    preds = []
    offsets = []
    seq_ids = []
    lbls = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl):
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            labels = batch[2].to(DEVICE)
            offset_mapping = batch[3]
            sequence_ids = batch[4]
            logits = model(input_ids, attention_mask)
            preds.append(logits.cpu().numpy())
            offsets.append(offset_mapping.numpy())
            seq_ids.append(sequence_ids.numpy())
            lbls.append(labels.cpu().numpy())
    
    # Concatenate results from all batches
    preds = np.concatenate(preds, axis=0)
    offsets = np.concatenate(offsets, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)
    lbls = np.concatenate(lbls, axis=0)
    
    # Get location predictions from model outputs
    location_preds = data_preprocess.get_location_predictions(preds, offsets, seq_ids, test=False)
    
    # Calculate and print the evaluation metrics
    score = data_preprocess.calculate_metrics(location_preds, offsets, seq_ids, lbls)
    print(score)
