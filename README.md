# DLNLP_assignment_24-SN23071847
This project uses the BERT base model(uncased) to complete the task of Score Clinical Patient Notes on Kaggle competition: https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes

## 1. Prerequisites
Initially, you need to download the datasets from Kaggle. Please check the README.md file in the Datasets folder for further information.
Next, you should create a conda environment and install the required using the command: 
```
conda env create -f environments.yml
```

## 2. Implementation
Just run the main.py file to see the output

## 3. Outputs
During running the main.py file, text.csv will be created which includes the text dataset. The nbme.pth file created saves the best model during training including the model weights. The first plot labelled as distribution of token lengths is used to determine the max_length hyperparameters and the second plot named Training and Validation Loss over Epochs indicates the loss change during training which is useful in ensuring convergence time. Finally the printed-out score contains precision, recall, F1 score and test accuracy that are used to analyze model performance.
