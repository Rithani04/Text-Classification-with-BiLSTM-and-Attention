# 🔍 Prompt Safety Classification using BiLSTM and Attention

## 📖 Project Overview
This project implements a multi-class text classification system for detecting unsafe and manipulated prompts. Three different datasets containing safe prompts, prompt injection attempts, jailbreak prompts, and policy-violating prompts are merged into a single unified dataset.

The goal of the project is to study how different neural architectures perform on prompt safety classification. A Bidirectional LSTM (BiLSTM) model is first trained as a baseline. The model is then enhanced by adding an Attention mechanism, and the performance difference between the two approaches is analyzed.

---

## 📂 Datasets Used
The final dataset is created by combining three independent datasets:

1. Prompt Injection Dataset  
   Source: deepset/prompt-injections  
   This dataset contains both safe prompts and prompt injection attempts.

2. Jailbreak Prompt Dataset  
   Source: TrustAIRLab/in-the-wild-jailbreak-prompts  
   This dataset contains real-world jailbreak prompts collected from public sources.

3. Policy Violating Prompt Dataset  
   This dataset contains prompts that violate safety or usage policies.

All datasets are converted into Pandas DataFrames and standardized into a common structure before merging.

---

## 🏷️ Classification Labels
The task is formulated as a four-class classification problem:

- Label 0: Safe Prompt  
- Label 1: Prompt Injection  
- Label 2: Jailbreak Prompt  
- Label 3: Policy Violating Prompt  

---

## 🔧 Data Preprocessing
The following preprocessing steps are applied before training:

- Loading datasets using the Hugging Face datasets library  
- Renaming columns to a unified format (text, label)  
- Merging all datasets into a single DataFrame  
- Handling class imbalance using resampling techniques and class weights  
- Tokenizing text using the Keras Tokenizer  
- Padding sequences to a fixed maximum length  
- Splitting the dataset into training and testing sets  

---

## 🧠 Model 1: Bidirectional LSTM (Baseline)

### Architecture Description
The baseline model consists of the following components:

- An Embedding layer that converts tokens into dense 128-dimensional vectors and ignores padded tokens using masking
- A Bidirectional LSTM layer with 128 hidden units to capture contextual information from both past and future tokens
- A pooling layer that reduces the sequence output into a fixed-size vector
- A Dense output layer with Softmax activation for four-class classification

### Rationale
Bidirectional LSTMs are effective for text classification tasks because they process sequences in both forward and backward directions. This allows the model to understand full sentence context, making it a strong baseline for prompt classification.

---

## 🧠 Model 2: BiLSTM with Attention

### Architecture Description
The second model extends the baseline architecture by introducing an Attention mechanism:

- An Embedding layer identical to the baseline model
- A Bidirectional LSTM layer with return_sequences enabled
- An Attention layer that learns to assign importance weights to each token in the sequence
- A context vector generated as a weighted sum of BiLSTM outputs
- A Dense Softmax output layer for classification

### Rationale
The Attention mechanism enables the model to focus on semantically important tokens instead of treating all words equally. This is particularly useful for detecting subtle manipulation patterns found in jailbreak and prompt injection prompts.

---

## ⚙️ Training Configuration
Both models are trained using the same configuration to ensure a fair comparison:

- Loss function: Categorical Crossentropy  
- Optimizer: Adam  
- Evaluation metric: Accuracy  
- Class weights: Applied to handle dataset imbalance  

---

## 📊 Results and Comparison
The BiLSTM with Attention model demonstrates improved performance compared to the baseline BiLSTM model. The Attention-based model achieves better accuracy and recall, especially for unsafe prompt categories.

The comparison highlights how Attention helps the model prioritize critical words and phrases that indicate prompt manipulation.

---

## 🛠️ Tech Stack
- Programming Language: Python  
- Deep Learning Framework: TensorFlow and Keras  
- Data Processing: Pandas and NumPy  
- Dataset Handling: Hugging Face datasets  
- Evaluation: scikit-learn
