#!/usr/bin/env python
# coding: utf-8

#pip install nltk
from google.colab import drive
drive.mount('/content/drive')

#pip install imblearn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
import re
import string

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize,sent_tokenize

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

df = pd.read_csv('/content/drive/MyDrive/NLP_capstone/Aerotek_Resume_Classification.csv')
df.head()

df['Category'].value_counts()
df.drop_duplicates(inplace = True)
df['Category'].value_counts()

df.head()

# data cleaning

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z]',' ',text)
    text = re.sub('http\S+',' ',text,flags=re.MULTILINE)   # to clean the http and https links,for multiple sentences (if review has many sentences) as well.
    text = re.sub('\.+',' ',text)                          # remove ..... if any
    word_tokens = word_tokenize(text)
    text = [words for words in word_tokens if words not in stops]   # clean text
    text = [words for words in text if words not in string.punctuation]  # to remove any punctuation marks
    return ' '.join(text)

df['cleaned_resume'] = df['Resume'].apply(lambda x: clean_text(x))
df.head()
df.drop('Resume',axis=1,inplace=True)
df = df[['cleaned_resume','Category']]
df.head()

df = df.sample(frac=1.0)
df['Category'].value_counts()
Category_encoded = {'Data Science':0,'Non-IT':1,'Testing':2,'Technical Maintenance':3,'Developer':4,'Database':5}
df['Category'] = df['Category'].map(Category_encoded)
df.head()

x = df['cleaned_resume']
y = df['Category']
x.head()
y.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 100)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_v = TfidfVectorizer()

tfidf_fit_trans_train = tfidf_v.fit_transform(x_train)
tfidf_fit_trans_train = tfidf_fit_trans_train.toarray()
tfidf_fit_trans_train.shape

x_test.shape
y_test.shape


tfidf_fit_trans_test = tfidf_v.transform(x_test)
tfidf_fit_trans_test = tfidf_fit_trans_test.toarray()
tfidf_fit_trans_test.shape

word_size = tfidf_fit_trans_train.shape[1]
word_size

vocab_size = word_size + 1
embedding_dimension = 300

lstm_model = Sequential([

    tf.keras.layers.Embedding(vocab_size,embedding_dimension),
    tf.keras.layers.LSTM(embedding_dimension),
    tf.keras.layers.Dense(embedding_dimension,activation = 'relu'),
    #tf.keras.layers.Dense(350,activation = 'relu'),
    tf.keras.layers.Dense(6,activation = 'softmax')
        
])

lstm_model.summary()

opt = tf.keras.optimizers.Adam(0.01) 

lstm_model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics= ['accuracy'])

num_epochs = 3

history = lstm_model.fit(tfidf_fit_trans_train,y_train,validation_data=(tfidf_fit_trans_test,y_test),epochs=num_epochs,batch_size = 100)

lstm_model.save('LSTM_Model_resume.h5')

test_data = ['data science machine learning computer vision']

tfidf_FT_test = tfidf_v.transform(test_data)
tfidf_FT_test = tfidf_FT_test.toarray()
tfidf_FT_test

new_model = tf.keras.models.load_model('./LSTM_Model_resume.h5')
new_model.summary()

pred = new_model.predict(tfidf_FT_test)
pred #softmax

labels = list(Category_encoded.keys())
labels

labels[np.argmax(pred)]

#Applying SMOTE

sm = SMOTE(random_state = 2)

X_resample,y_resample=sm.fit_sample(tfidf_fit_trans_train,y_train.values.ravel())
X_train1.shape
y_train1.shape

X_test1.shape
y_test1.shape

y_resample=pd.DataFrame(y_resample)
X_resample=pd.DataFrame(X_resample)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_resample, y_resample, test_size = 0.3, random_state=100)

word_size = X_train1.shape[1]
word_size

vocab_size = word_size + 1
embedding_dimension = 300

lstm_model = Sequential([

    tf.keras.layers.Embedding(vocab_size,embedding_dimension),
    tf.keras.layers.LSTM(embedding_dimension),
    tf.keras.layers.Dense(embedding_dimension,activation = 'relu'),
    #tf.keras.layers.Dense(350,activation = 'relu'),
    tf.keras.layers.Dense(6,activation = 'softmax')
        
])

lstm_model.summary()

opt = tf.keras.optimizers.Adam(0.01) 

lstm_model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics= ['accuracy'])

num_epochs = 3

history = lstm_model.fit(X_train1,y_train1,validation_data=(X_test1,y_test1),epochs=num_epochs,batch_size = 100)

lstm_model.save('LSTM_Model_resume.h5')

test_data = ['data science machine learning computer vision']

tfidf_FT_test1 = tfidf_v.transform(test_data)
tfidf_FT_test1 = tfidf_FT_test1.toarray()
tfidf_FT_test1

new_model = tf.keras.models.load_model('./LSTM_Model_resume.h5')
new_model.summary()

pred = new_model.predict(tfidf_FT_test1)
pred #softmax

labels = list(Category_encoded.keys())
labels

labels[np.argmax(pred)]
