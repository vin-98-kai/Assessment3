# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:13:44 2022

@author: Calvin
"""

import os
import re
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules_for_articles_cate import ModelCreation

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

#Static
CSV_URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
OHE_PICKLE_PATH = os.path.join(os.getcwd(),'models','ohe_fname.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'models','model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'json','tokenizer_category.json')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(), 'logs',log_dir)
vocab_size = 500
oov_token = 'OOV'
max_len = 333 #got this value by rounding off np.median(length_of_text)

# EDA
# Step 1) Data Loading
df = pd.read_csv(CSV_URL)

# Step 2) Data Inspection
df.head(10)
df.info()

df['category'].unique() # to get the unique targets
df['text'][0]
df['category'][0]
df.duplicated().sum() # 99 duplicates
df[df.duplicated()] # check where the duplicates loc

# Step 3) Data Cleaning
df = df.drop_duplicates() # Remove duplicates
text = df['text'].values # Features : X
category = df['category'].values # category : y

for index,tex in enumerate(text):
    text[index] = re.sub('(.*?)',' ',tex)
    text[index] = re.sub('[^a-zA-Z]',' ',tex).lower().split()

# Step 4) Features Selection (Dont have)

# Step 5) Data Preprocessing
# tokenization
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
# print(word_index)

# to convert into numbers
train_sequences  = tokenizer.texts_to_sequences(text)

# Padding & Truncating
length_of_text = [len(i) for i in train_sequences] # list comprehension
print(np.median(length_of_text)) # to get the number of maxlength for padding

padded_text = pad_sequences(train_sequences,maxlen=max_len,
                              padding='post',
                              truncating='post')

# One Hot Encoding for the Target
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))
with open(OHE_PICKLE_PATH,'wb') as file:
    pickle.dump(ohe,file)

# Train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_text,category,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model Development
mc = ModelCreation()
model = mc.bidirection_lstm_layer(X_train,num_node=64)

plot_model(model,show_layer_names=(True),show_shapes=(True))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

# callbacks
tensorboard_callback=TensorBoard(log_dir=LOG_FOLDER_PATH)
early_stopping_callback = EarlyStopping(monitor='loss',patience=5)

hist = model.fit(X_train,y_train,
          epochs=70,
          batch_size=128,
          validation_data=(X_test,y_test),
          callbacks=[tensorboard_callback,early_stopping_callback])

#%%
hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation loss')
plt.legend()
plt.title('Value Loss')
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--',label='Training loss')
plt.plot(hist.history['val_acc'],label='Validation loss')
plt.legend()
plt.title('Value Accuracy')
plt.show()

#%% Model Evaluation
y_true = y_test
y_pred = model.predict(X_test)

y_true = np.argmax(y_true,axis=1)
y_pred = np.argmax(y_pred,axis=1)

print(classification_report(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

#%% Model Saving
model.save(MODEL_SAVE_PATH)
    
token_json = tokenizer.to_json()

with open(TOKENIZER_PATH,'w') as file:
        json.dump(token_json,file)

#%% Discussion / Report
# Model achieved around 80% accuracy during training without EarlyStopping
# Recall and f1 score reports 78% and 78%% respectively without EarlyStopping
# However the model hit 35% after applying EarlyStopping
# Need more data because most of them is padded and embedded
# Flatten() maybe introduced during training
# decrease dropout rate to control overfitting
# Trying with different DL architecture for example BERT model, transformer
# model, GPT3 model may help to improve the model