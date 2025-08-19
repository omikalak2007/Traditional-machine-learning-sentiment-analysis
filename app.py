import streamlit as st
import pickle

# Load BoW vectorizer and trained model
with open("BOW_vectoriser.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("sentiment_analysis_ml_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a movie related sentence to analyze sentiment:")

# User input
user_input = st.text_area("Your text here")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform text using BoW
        text_vector = vectorizer.transform([user_input])

        # Predict using model
        prediction = model.predict(text_vector)[0]
        if prediction == 1:
            prediction_text = "Positive"
        else:
            prediction_text = "Positive"

        # Show result
        st.success(f"Predicted Sentiment: {prediction_text}")
