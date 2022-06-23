# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:38:33 2022

@author: Calvin
"""

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Bidirectional,Embedding

#%% Functions

class ModelCreation():
    def __init__(self):
        pass
    
    def bidirection_lstm_layer(self,X_train,num_node=128,dropout_rate=0.2,
                          output_node=5,vocab_size=500):
        embedding_dim = 64
        model = Sequential()
        model.add(Input(shape=(333)))
        model.add(Embedding(vocab_size,embedding_dim))
        model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(num_node))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_node,activation=('relu')))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_node,'softmax'))
        model.summary()
        
        return model