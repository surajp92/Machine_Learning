#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:13:36 2019

@author: user1
Link: https://www.youtube.com/watch?v=iMIWee_PXl8&list=PLVBorYCcu-xWahQ0u2_guKSJ-0fc8VKb2&index=7
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

leng = 3

# data for training
data = [[i+j for j in range(leng)] for i in range(80)]
data = np.array(data, dtype=np.float32)
target = [[i+j+1 for j in range(leng)] for i in range(1,81)]
target = np.array(target, dtype=np.float32)

data = data.reshape(80,1,leng)/200
target = target.reshape(80,1,leng)/200

# validation data
data_val = [[i+j for j in range(leng)] for i in range(80,100)]
data_val = np.array(data_val, dtype=np.float32)
target_val = [[i+j+1 for j in range(leng)] for i in range(81,101)]
target_val = np.array(target_val, dtype=np.float32)

data_val = data_val.reshape(20,1,leng)/200
target_val = target_val.reshape(20,1,leng)/200

# test data

# build model
model = Sequential()
model.add(LSTM(leng, input_shape=(1, leng), return_sequences=True, activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng), return_sequences=True, activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng), return_sequences=True, activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng), return_sequences=True, activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng), return_sequences=True, activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng), return_sequences=True, activation='sigmoid'))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(data, target, nb_epoch=10000, batch_size=50, validation_data=(data_val, target_val))

predict = model.predict(data)

# plotting results
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validationloss')
plt.legend()
plt.show()
