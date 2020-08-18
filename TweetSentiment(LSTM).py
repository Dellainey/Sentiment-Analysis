# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 09:02:58 2018

@author: Dellainey
"""

#LSTM model for sentiment analysis of tweets
#importing the required libraries and functions
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding
from keras.layers import Activation
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import re
#importing the dataset
data = pd.read_csv('../input/training.1600000.processed.noemoticon.csv',encoding='latin-1',names = ['target','one','two','three','four','text'])
#Scaling down the dataset 
rv = np.random.rand(data.shape[0])
data = data.loc[rv<0.02]
# keeping only the columns of interest
data = data[['target','text']]
print(data[data['target']==0].size)
print(data[data['target']==4].size)
#Tokenizing the words from the tweets. Considering only top 1000 common words. This number can be changed
tokenizer = Tokenizer(num_words=1000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
tokenizer.fit_on_texts(data['text'].values)
x = tokenizer.texts_to_sequences(data['text'].values)
#Padding the squences with 0s whereever needed
text = pad_sequences(x)
# Building the LSTM model
model = Sequential()
#input_dim = 1000 and output_dim = 140. These values can be changed
model.add(Embedding(1000,140,input_length = text.shape[1]))
model.add(LSTM(units=140, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
model.add(Dense(5,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# converting the format of the target to categorical
target = data['target']
target = to_categorical(target)
#Splitting the data into train and test sets. 80-20 partition
train_text, test_text, train_target, test_target = train_test_split(text,target, train_size = 0.80,random_state = 12345)
print(train_text.shape,test_text.shape)
print(train_target.shape,test_target.shape)
# running the model on the training set
model.fit(train_text, train_target, epochs = 7, batch_size = 10, verbose = 2)
#Evaluating the model on the test set
score = model.evaluate(test_text, test_target, verbose = 2)
print("Score := " + str(score[0]))
print("Accuracy := " + str(score[1]))