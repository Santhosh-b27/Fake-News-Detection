# Fake News Classification Using BERT

This project implements a Fake News Classification System using BERT (Bidirectional Encoder Representations from Transformers). It fine-tunes a pre-trained BERT model to classify news articles as real or fake based on textual content.

## 📌 Key Steps in the Code

### 1. Install Required Libraries (Commented Out)

Uses Transformers, Torch, Datasets, Pandas, and Scikit-learn for model training and evaluation.

### 2. Load Dataset

A small dataset is created with three sample news headlines labeled as real (1) or fake (0).

Converts labels from categorical to numerical values.

### 3. Load BERT Tokenizer

Uses BertTokenizer from Hugging Face’s transformers library to tokenize the text.

### 4. Define a Custom Dataset Class (NewsDataset)

Tokenizes text, applies padding, truncation, and returns tensors.

Stores encoded inputs and corresponding labels for PyTorch DataLoader.

### 5. Split Data into Train and Test Sets

Uses train_test_split to create training (80%) and testing (20%) datasets.

Converts text and labels into NewsDataset format.

### 6. Load Pre-trained BERT Model

Loads bert-base-uncased model for sequence classification with two output labels (real/fake news).

Sets the model to training mode.

### 7. Define Training Parameters

Uses AdamW optimizer with a learning rate of 5e-5.

Creates a DataLoader with batch size 8 for training.

### 8. Train the Model (2 Epochs)

Moves model to GPU (if available) or CPU.

Performs forward propagation, loss computation, backpropagation, and optimizer update.

### 9. Evaluate the Model

Moves the model to evaluation mode.

Uses DataLoader to process test data in batches.

Computes accuracy and prints a classification report.

### 10. Predict Fake or Real News

Defines a function predict_news(news).

Tokenizes the input, passes it through BERT, and predicts Real or Fake.

Provides an example prediction for a sample news headline.

