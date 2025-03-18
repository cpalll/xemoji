import joblib

# Load the vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('logistic_regression_model.pkl')


def preprocess_text(text: str) -> str:
    # Preprocess the text
    text = text.lower()

    return text


# Predict emojis for new text
new_text = "I LOVE MATHEMATICS"
preprocessed_text = preprocess_text(new_text)  # Preprocess the text
X_new = vectorizer.transform([preprocessed_text])  # Transform using the vectorizer
predicted_emoji = model.predict(X_new)[0]  # Predict the emoji
print(f"Predicted Emoji: {predicted_emoji}")