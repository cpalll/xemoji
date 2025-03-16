import pandas as pd
import numpy as np

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 10)       # Limit rows displayed
pd.set_option('display.width', 1000)        # Adjust display width
pd.set_option('display.max_colwidth', 50)   # Limit column width

tweets = pd.read_csv('twitter_sentiment.csv', encoding='latin-1')
tweets.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']  # Rename columns

# Add index column
tweets['index'] = range(0, len(tweets))
tweets.set_index('index')
# Remove all columns except for sentiment and text
tweets = tweets[['index', 'sentiment', 'text']]


print(tweets.head())