# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 20:47:01 2018

@author: Dellainey
"""
#Logistic Regression model for sentiment analysis of tweets
#importing the required libraries and functions
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import re
#importing the dataset
data = pd.read_csv('../input/training.1600000.processed.noemoticon.csv',encoding='latin-1',names = ['target','one','two','three','four','text'])
# Scaling down the dataset to 2 percent of the original
rv = np.random.rand(data.shape[0])
data = data.loc[rv<0.02]
# replacing the target value from 4 to 1 
data['target'] = np.where(data['target']== 4, 1, data['target'])
print(data['target'].value_counts())
# cleaning the text column. using regular expression. takes out everything except numbers from 0 to 9,
#alphabets lower and uppercase and keeping spaces
data['text'] =data['text'].apply(lambda x: re.sub('[^0-9a-zA-Z\s]','',x))
data.head(5)
#Tokenizing the words before stemming them
data['text'] = data.apply(lambda row: word_tokenize(row['text']),axis=1)
# Stemming the words from the tweets using PorterStemmer
ps = PorterStemmer()
data['text'] = data['text'].apply(lambda row: [ps.stem(word)for word in row])
# joining back the words to form a sequence
data['text'] = data['text'].apply(lambda row: " ".join(row))
print(data.head(5))
# Vectorizing the text
vectorizer = CountVectorizer(lowercase=False)
x = vectorizer.fit_transform(data['text'])
target = data['target']
#Spliting the dataset into train and test sets
train_text, test_text, train_target, test_target = train_test_split(x,target, train_size = 0.80,random_state = 12345)
print(train_text.shape,test_text.shape)
print(train_target.shape,test_target.shape)
# building a logistic regression model
LG_model = LogisticRegression()
LG_model = LG_model.fit(train_text,train_target)
LG_pred = LG_model.predict(test_text)
#Calculating the accuracy of the model
print(accuracy_score(test_target,LG_pred))

