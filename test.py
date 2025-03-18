import joblib

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')


def preprocess_text(text: str) -> str:
    # Preprocess the text
    text = text.lower()
    # Remove punctuation
    text = lambda x: re.sub(r'[^\w\s]', '', x)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    # Tokenize text
    text = text.apply(word_tokenize)

    return text


# Predict emojis for new text
new_text = "I LOVE MATHEMATICS"
preprocessed_text = preprocess_text(new_text)  # Preprocess the text
X_new = vectorizer.transform([preprocessed_text])  # Transform using the vectorizer
predicted_emoji = model.predict(X_new)[0]  # Predict the emoji
print(f"Predicted Emoji: {predicted_emoji}")