import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib




nltk.download('stopwords')

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 10)       # Limit rows displayed
pd.set_option('display.width', 1000)        # Adjust display width
pd.set_option('display.max_colwidth', 50)   # Limit column width

tweets = pd.read_csv('twitter_sentiment.csv', encoding='latin-1')
tweets.columns = ['text', 'sentiment']  # Rename columns

#Add index column
tweets['index'] = range(0, len(tweets))
tweets.set_index('index', inplace=True)

# Remove all columns except for sentiment and text
tweets = tweets[['sentiment', 'text']]
#Assign emoji to corresponding sentiment value
tweets['sentiment_emoji'] = tweets['sentiment'].map({-1: '😠', 0: '😐', 1: '😀'})

#Preprocess the text
tweets['text'] = tweets['text'].str.lower()
#Remove missing value rows
tweets.dropna(inplace=True)
#Remove punctuation
tweets['text'] = tweets['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))


stop_words = set(stopwords.words('english'))
tweets['text'] = tweets['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
#Tokenize text
tweets['text'] = tweets['text'].apply(word_tokenize)

#Convert text into numerical features for TF-IDF
# Use TF-IDF vectorizer
tweets['text'] = tweets['text'].apply(lambda x: ' '.join(x))
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(tweets['text'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, tweets['sentiment_emoji'],
                                                    test_size=0.2, random_state=42)

# Save the vectorizer and processed data
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump((X_train, X_test, y_train, y_test), 'processed_data.pkl')


print(tweets.head())

