#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:17:27 2019

@author: user1
Link: https://github.com/shreyans29/thesemicolon/blob/master/lstm%20-%20RNN.py
"""
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

data = [[i for i in range(100)]]
data = np.array(data, dtype=float)
target = [[i for i in range(1,101)]]
target = np.array(target, dtype=float)

data = data.reshape((1,1,100))
target = target.reshape((1,1,100))

x_val = [i for i in range(100,200)]
x_val = np.array(x_val).reshape((1,1,100))
y_val = [i for i in range(101,201)]
y_val = np.array(y_val).reshape((1,1,100))

x_test = [i for i in range(200,300)]
x_test = np.array(x_test).reshape((1,1,100))
# y_test = [i for i in range(101,201)]
# y_test = np.array(y_test).reshape((1,1,100))

# create the model
model = Sequential()
model.add(LSTM(100, input_shape=(1,100), return_sequences=True))
model.add(Dense(100))

# compile the model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(data, target, nb_epoch=10000, batch_size=1, verbose=2, validation_data=(x_val, y_val))

predict = model.predict(data)
# predict = model.predict(6)

# plotting results
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()

plt.show()