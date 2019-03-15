#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:33:09 2019

@author: user1
code to solve ODEs using LSTM 
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

omega = 2.0*np.pi
T = 1.0
dt = 0.01
nt = int(T/dt)

# data = y1 and y2; target = slope = (y^(n+1)-y^n)/dt
# y1(t) = cos(wt), y2(t) = -w*sin(wt)
data = [[np.cos(omega*dt*i), -omega*np.sin(omega*dt*i)] for i in range(nt)]
data = np.array(data, dtype=np.float64)
data = data/omega

target = [[(data[i,0]-data[i-1,0])/dt, (data[i,1]-data[i-1,1])/dt ] for i in range(1,nt)]
target = np.array(target, dtype=np.float64)
target = target/omega

# data for training

data_train = data[1:100].reshape(99,1,2)
target_train = target.reshape(99,1,2)

# build model
model = Sequential()
#model.add(LSTM(2, input_shape=(1, 2), return_sequences=True, activation='relu'))
#model.add(LSTM(2, input_shape=(1, 2), return_sequences=True, activation='relu'))
#model.add(LSTM(2, input_shape=(1, 2), return_sequences=True, activation='relu'))
#model.add(LSTM(2, input_shape=(1, 2), return_sequences=True, activation='relu'))
#model.add(LSTM(2, input_shape=(1, 2), return_sequences=True, activation='relu'))
#model.add(LSTM(2, input_shape=(1, 2), return_sequences=True, activation='relu'))

model.add(LSTM(100, input_shape=(1,2), return_sequences=True, activation='relu'))
model.add(Dense(2))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(data_train, target_train, nb_epoch=8000, batch_size=50, validation_split=0.2)

# plotting results
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# deployment

test = [1.0, 0.0]
test = np.array(test)
test = test.reshape(1,1,2)

temp = [1.0, 0.0]
temp= np.array(temp)
temp = test.reshape(1,1,2)

for i in range(1,nt):
    test = model.predict(test)
    a = temp[i-1][0][0] + test[0][0][0]*dt
    b = temp[i-1][0][1] + test[0][0][1]*dt
    c = [a,b]
    c = np.array(c)
    c = c.reshape(1,1,2)    
    temp = np.vstack((temp, c))
    





















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
