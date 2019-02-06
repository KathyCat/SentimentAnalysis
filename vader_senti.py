# use vader to analyze twitter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
import pandas as pd
import re
import csv


blimit = 50
sentiment_analyzer = SentimentIntensityAnalyzer()

                         
# read tweets from CSV file
for chunk in pd.read_csv('batch.csv', encoding ="utf-8", chunksize=blimit): 
    tweets = chunk['Tweet']
    # preprocess tweets
    
    for tweet in tweets_ori:
        
        tweet = re.sub( r'\S*/\S*', '', tweet, re.M|re.I) # delete urls
        tweet = re.sub( r'&.*;', '', tweet, re.M|re.I)
        tweet = re.sub( r'\bRT\b', '', tweet, re.M|re.I)
        tweet = re.sub( r'@\w*:?', '', tweet, re.M|re.I)
        if re.search(r'[a-zA-Z]', tweet) is None:
            continue
        
        if detect(tweet) == 'en': # delete non-english tweets
            tweets.append(tweet)
    
    
     
    # give each twitter with sentiment score
    rows = []
    for tweet in tweets: 
        score = sentiment_analyzer.polarity_scores(tweet)
        neg = score.get('neg')
        com = score.get('compound')
        pos = score.get('pos')
        # not append tweet with score (0,0,0)
        if com == 0 and neg == 0 and pos == 0:
            continue
        
        row = [tweet]
        row.append(neg)
        row.append(com)         
        row.append(pos)  
        row.append(len(tweet.split()))
        
        rows.append(row)

        
    with open("batch_result.csv", "a", encoding='utf-8', newline='') as file: # newline='' to avoid empty line 
        writer = csv.writer(file)    
        writer.writerows(rows) 
         



        
    
    
           
