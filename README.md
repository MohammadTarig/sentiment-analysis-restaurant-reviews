# Sentiment Analysis of Restaurant Review

A BERT-based NLP project that classifies restaurant reviews as **Positive, Neutral, or Negative**.

The project includes a complete training workflow in Jupyter Notebook and a Streamlit web app for making predictions on new reviews.

## 📌 Overview

The model was fine-tuned from `bert-base-uncased` using the [**10,000 Restaurant Reviews** dataset from Kaggle](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews).

Since the dataset does not contain sentiment labels, they were created from the ratings:

|Rating|Sentiment|
|---|---|
|< 3|Negative|
|= 3|Neutral|
|> 3|Positive|

After cleaning, **9,954 reviews** were used for training and evaluation.

### Dataset Distribution

- 🟢 Positive: 6,315 (63.44%)
    
- 🔴 Negative: 2,447 (24.58%)
    
- 🟡 Neutral: 1,192 (11.98%)
    

## 🤖 Model

- **Model:** `bert-base-uncased`
    
- **Task:** 3-class text classification
    
- **Maximum sequence length:** 128
    
- **Training epochs:** 3
    
- **Batch size:** 16
    
- **Learning rate:** 2e-5
    
- **Split:** 80% train / 10% validation / 10% test
    
- **Training:** Google Colab with Tesla T4 GPU
    

## 📊 Results

Evaluation was performed on a held-out test set of **996 reviews**.

|Metric|Score|
|---|--:|
|Accuracy|**87%**|
|Weighted F1|**0.85**|
|Macro F1|**0.72**|

|Class|Precision|Recall|F1|
|---|--:|--:|--:|
|Negative|0.81|0.88|0.84|
|Neutral|0.59|0.27|0.37|
|Positive|0.91|0.97|0.94|

The model performs well on Positive and Negative reviews but struggles with Neutral reviews. This is the main limitation of the project and is partly related to the rating-based labeling approach and class imbalance.

## 🌐 Streamlit App

The application allows users to enter a restaurant review and receive:

- Sentiment prediction
    
- Prediction confidence
    
- Probabilities for all three classes
    

### Running the Application

Clone the repository and navigate to the project directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The trained model is downloaded from Hugging Face when the application loads.

## 🤗 Trained Model

The fine-tuned model is available on Hugging Face:

[mohammad-tarig/bert_base_uncased_sentiment_restaurants_v1](https://huggingface.co/mohammad-tarig/bert_base_uncased_sentiment_restaurants_v1/?utm_source=chatgpt.com)

## 🛠️ Technologies

**Python · PyTorch · Hugging Face Transformers · Scikit-learn · Pandas · NumPy · Streamlit**

## ⚠️ Limitations

- Sentiment labels were derived from ratings rather than manually annotated.
    
- The dataset is imbalanced.
    
- Neutral sentiment has substantially lower performance.
    
- Reviews longer than 128 tokens are truncated.
    
- The model is specifically trained on restaurant reviews.