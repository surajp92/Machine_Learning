#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:47:58 2019

@author: Suraj Pawar
"""
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import matplotlib.pyplot as plt
import pandas as pd
from create_data import * #create_training_data_bdf2
from model_prediction import *

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

dataset_train = pd.read_csv('./a.csv', sep=",",skiprows=0,header = None, nrows=1000)
m,n=dataset_train.shape
training_set = dataset_train.iloc[:,0:n].values

dt = training_set[1,0] - training_set[0,0]
training_set = training_set[:,1:n]

#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range=(0,1))
#training_set_scaled = sc.fit_transform(training_set)
#training_set_scaled.shape
#training_set = training_set_scaled

legs = 2
slopnet = "BDF"
xtrain, ytrain = create_training_data(training_set, m, n, dt, legs, slopnet)

from keras.models import Sequential, Model
from keras.layers import Dense, dot, Input
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers

# create the LSTM model
model = Sequential()

# Layers start
input_layer = Input(shape=(legs*(n-1),))

# Hidden layers
x = Dense(100, activation='tanh', use_bias=True)(input_layer)
x = Dense(100, activation='tanh', use_bias=True)(x)
#x = Dense(100, activation='tanh', use_bias=True)(x)
#x = Dense(100, activation='relu', use_bias=True)(x)
#x = Dense(100, activation='relu', use_bias=True)(x)

op_val = Dense(10, activation='linear', use_bias=True)(x)
custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
history_callback = custom_model.fit(xtrain, ytrain, epochs=400, batch_size=100, verbose=1, validation_split= 0.2,
                                    callbacks=callbacks_list)

# training and validation loss. Plot loss
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]
# evaluate the model
scores = custom_model.evaluate(xtrain, ytrain)
print("\n%s: %.2f%%" % (custom_model.metrics_names[1], scores[1]*100))

epochs = range(1, len(loss_history) + 1)

plt.figure()
plt.plot(epochs, loss_history, 'b', label='Training loss')
plt.plot(epochs, val_loss_history, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#read data for testing
dataset_test = pd.read_csv('./a.csv', sep=",",header = None, skiprows=1000)
dataset_total = pd.concat((dataset_train,dataset_test),axis=0)
dataset_total.drop(dataset_total.columns[[0]], axis=1, inplace=True)
m,n=dataset_test.shape
testing_set = dataset_test.iloc[:,0:n].values
testing_set = testing_set[:,1:n]

# predict the 
ytest_ml = model_predict(testing_set, m, n, dt, legs, slopnet)


for i in range(n-1):
    plt.figure()    
    plt.plot(ytest_ml[:,i], 'b-', label=r'$y_'+str(i+1)+'$'+'ML') 
    plt.plot(testing_set[:,i], 'r-', label=r'$y_'+str(i+1)+'$') 
    plt.ylabel('response ML')
    plt.xlabel('time ML')
    plt.legend(loc='best')
    name = 'dns_bdf_twoleg_a='+str(i+1)+'.eps'
    #plt.savefig(name, dpi = 600)
    plt.show()
