import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path


# -----------------------------------
# Configuration
# -----------------------------------

MODEL_NAME = "mohammad-tarig/bert_base_uncased_sentiment_restaurants_v1"

id2label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# -----------------------------------
# Load trained model
# -----------------------------------

@st.cache_resource
def load_model():

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# -----------------------------------
# Prediction
# -----------------------------------

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    prediction = torch.argmax(probabilities, dim=-1).item()

    sentiment = id2label[prediction]

    confidence = probabilities[0][prediction].item()

    return sentiment, confidence, probabilities[0].numpy()


# -----------------------------------
# Streamlit UI
# -----------------------------------

st.set_page_config(
    page_title="Restaurant Reviews Sentiment Analyzer",
    page_icon="🍽️"
)

st.title("🍽️ Restaurant Reviews Sentiment Analyzer")

st.write(
    "Enter a restaurant review and the BERT model "
    "will classify it as Positive, Neutral, or Negative."
)

review = st.text_area(
    "Restaurant Review",
    height=150,
    placeholder="Example: The food was delicious and the service was excellent!"
)

if st.button("Analyze Sentiment"):

    if not review.strip():

        st.warning("Please enter a restaurant review.")

    else:

        sentiment, confidence, probabilities = predict_sentiment(review)

        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success(f"🟢 Positive — {confidence:.1%} confidence")

        elif sentiment == "Negative":
            st.error(f"🔴 Negative — {confidence:.1%} confidence")

        else:
            st.warning(f"🟡 Neutral — {confidence:.1%} confidence")

        st.subheader("Class Probabilities")

        st.write(f"🔴 Negative: {probabilities[0]:.1%}")
        st.write(f"🟡 Neutral: {probabilities[1]:.1%}")
        st.write(f"🟢 Positive: {probabilities[2]:.1%}")