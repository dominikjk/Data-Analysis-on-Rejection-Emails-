# Rejection Email Analysis

A Python analysis of job application emails, comparing rejections with non-rejections to see how companies word bad news.

## What it looks at

- **Length**: how long rejection emails are compared with other replies
- **Lexical richness**: unique words divided by total words, averaged for each group
- **Sentiment**: compound scores from NLTK's VADER, plotted as a distribution for each group
- **Word associations**: a Word2Vec model (gensim) trained on the emails, used to find the words that appear in similar contexts to "developer"

## Tools

Python, pandas, NumPy, matplotlib, NLTK (VADER), gensim

The dataset (129 labelled emails: 65 rejections, 64 other replies) is included in the repo.
