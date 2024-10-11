import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv).
from gensim.models import Word2Vec
import nltk
import string
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer


df = pd.read_csv('Rejection Data - Sheet1.csv')
df.tail()

df.Status.value_counts().plot(kind='bar')
plt.show()

#Working out the length of every data set in df column
df['Length'] = df['Tokens'].apply(len)

#Visualising the length and status in df 
df.plot.scatter(x='Length', y='Status')
plt.title('email length vs status')
plt.xlabel('email length')
plt.ylabel('status')
plt.show()

#calculating the lexical richness for each email in the dataset
df['LexRich'] = df['Tokens'].apply(lambda x: len(set(x)) / len(x))

#taking step further and working out lexical richness for both reject and non reject emails
lex_reject = df[df.Status == 'reject']['LexRich'].mean()
lex_not_reject = df[df.Status == 'not_reject']['LexRich'].mean()

#training a Word2Vec model 
from gensim.models import Word2Vec

model = Word2Vec(sentences=df['Tokens'], vector_size=100, window=5, min_count=1, workers=4) #adding required parameters

similar_words = model.wv.most_similar(positive=['developer'], topn=20)

for word, similarity in similar_words:
  print(word, similarity)


#plotting a frequency chart for the distribution of sentiment scores
sid = SentimentIntensityAnalyzer()


def calculate_sentiment_score(text):
    return sid.polarity_scores(text)['compound']
    

#apply SA to each email and add new column for score
df['SentimentScore'] = df['Email'].apply(calculate_sentiment_score)

#plot distribution of sentiment scores for r and nonr emails
plt.hist(df[df['Status'] == 'reject']['SentimentScore'], bins=20, alpha=0.5, label='Reject')
plt.hist(df[df['Status'] == 'not_reject']['SentimentScore'], bins=20, alpha=0.5, label='Not Reject')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('distribution of sentiment scores')
plt.legend()
plt.show()
