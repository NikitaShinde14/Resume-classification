#!/usr/bin/env python
# coding: utf-8

#pip install nltk
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

df = pd.read_csv('Aerotek_Resume_Classification.csv')
df.head()

df['Category'].value_counts()

df.drop_duplicates(inplace = True)

df['Category'].value_counts()

DataScience = df['Category'] == 'Data Science'
df_try = df[DataScience]
df= df.append([df_try]*2,ignore_index=True)

Testing = df['Category'] == 'Testing'
df_try = df[Testing]
df= df.append([df_try]*2,ignore_index=True)

TechnicalMaintenance = df['Category'] == 'Technical Maintenance'
df_try = df[TechnicalMaintenance]
df= df.append([df_try]*1,ignore_index=True)

Database = df['Category'] == 'Database'
df_try = df[Database]
df= df.append([df_try]*1,ignore_index=True)
df


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

from sklearn.feature_extraction.text import CountVectorizer
countVect = CountVectorizer()

x_train_counts = countVect.fit_transform(x_train)

x_train_counts.toarray()
x_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer

Tf_idf_t = TfidfTransformer()

x_train_tfidf = Tf_idf_t.fit_transform(x_train_counts)
x_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
mn=MultinomialNB()

clf = mn.fit(x_train_tfidf,y_train)

# pipeline is code saving but optional

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SVMSMOTE


#text_clf_pipe = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('smote', SMOTE(random_state=12)),('clf',MultinomialNB())])
text_clf_pipe = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('smote', SMOTE(random_state=12)),('clf',MultinomialNB())])
text_clf_pipe = text_clf_pipe.fit(x_train,y_train)

predicted = text_clf_pipe.predict(x_test)
text_clf_pipe.score(x_test,y_test)
predicted
sample_test = ['Prioritizes and manages database programming and management, and CRF development activities for staff across multiple projects Works with senior management to plan resources and capacity Oversees the development of programs and scripts used to monitor, manage and analyze clinical data across all projects Manages projects according to core team timelines and provides input to those timelines Expert knowledge of database management systems and network architecture Proactively researches opportunities to utilize new technology to improve quality, productivity and efficiency Develops and enforces high-level SOPs for database activities']

predSample = text_clf_pipe.predict(sample_test)
predSample