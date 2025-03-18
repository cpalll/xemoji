# xemoji
Predicts emoji sentiment (😀, 😐, 😠) of a tweet using machine model trained on 160,000 tweets from the 
[Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset). 

# Features
- **Text Preprocessing and TF-IDF Vectorization**: Tokenization, stopwords removal, and lowercase conversion, before converting text into numerical features
- **Logistic Regression Model**: Predicts emojis with 89.03% accuracy.

# ☑️ TO-DO
- Deploy web-app to easily utilise model for sentiment analysis

# Model accuracy
              precision    recall  f1-score   support

           😀       0.92      0.89      0.90     14375
           😐       0.86      0.97      0.91     11067
           😠       0.88      0.77      0.82      7152

    accuracy                               0.89     32594
	macro avg          0.89      0.88      0.88     32594
	weighted avg       0.89      0.89      0.89     32594
