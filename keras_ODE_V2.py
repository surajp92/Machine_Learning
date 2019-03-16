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
from sklearn.preprocessing import MinMaxScaler


omega = 2.0*np.pi
T = 1.0
dt = 0.01
nt = int(T/dt)

# data = y1 and y2; target = slope = (y^(n+1)-y^n)/dt
# y1(t) = cos(wt), y2(t) = -w*sin(wt)
data = [[np.cos(omega*dt*i), -omega*np.sin(omega*dt*i)] for i in range(nt)]
data = np.array(data, dtype=np.float64)
data = data

target = [[(data[i,0]-data[i-1,0])/dt, (data[i,1]-data[i-1,1])/dt ] for i in range(1,nt)]
target = np.array(target, dtype=np.float64)
target = target

# normalization
scaler = MinMaxScaler(feature_range=(0, 1))
#data = scaler.fit_transform(data)
#target = scaler.fit_transform(target)


# data for training

# data_train = data[1:100].reshape(99,1,2)
# target_train = target.reshape(99,1,2)
data_train = data[1:100].reshape(99,1,2)
target_train = target #.reshape(1,99,2)

model = Sequential()
model.add(LSTM(4, input_shape=(1, 2), activation='tanh'))
model.add(Dense(2))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(data_train, target_train, nb_epoch=10000, batch_size=10, validation_split=0.2)

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

test = [2.0, 0.0]
test = np.array(test)
test = test.reshape(1,1,2)

temp = [2.0, 0.0]
temp= np.array(temp)
temp = temp.reshape(1,2)
# temp = scaler.inverse_transform(temp)

for i in range(1,nt):
    slope = model.predict(test)
    # slope = scaler.inverse_transform(slope)
    a = temp[i-1,0] + slope[i-1,0]*dt
    b = temp[i-1,1] + slope[i-1,1]*dt
    c = [a,b]
    c = np.array(c)
    temp = np.vstack((temp, c))
    # test2 = scaler.fit_transform(temp)
    c = c.reshape(1,1,2)    
    # test = test2
    test = np.vstack((test, c))
    

# temp = scaler.inverse_transform(temp)    
# slope for y1
plt.figure()
#plt.plot(predictData2[:,0], 'b', label='Prdicted slope')
plt.plot(temp[:,0], 'r', label='Predicted slope')
plt.title('Predicted  Y1')
plt.legend()

plt.show()

# slope for y2
plt.figure()
#plt.plot(predictData2[:,1], 'b', label='Prdicted slope')
plt.plot(temp[:,1], 'b', label='Predicted slope')
plt.title('Predicted Y2')
plt.legend()

plt.show()

true_y1 = [0.0 for i in range(nt)]
true_y2 = [0.0 for i in range(nt)]
true_y1= np.array(true_y1, dtype=np.float64)
true_y2= np.array(true_y2, dtype=np.float64)

true_y1[0] = 2.0
true_y2[0] = 0.0

for i in range(1,nt):
    true_y1[i] = true_y1[i-1] + dt * (-omega*np.sin(omega*dt*(i-1)))
    true_y2[i] = true_y2[i-1] + dt * (-omega*omega*np.cos(omega*dt*(i-1)))

# euler forward for y1    
plt.figure()
#plt.plot(predictData2[:,0], 'b', label='Prdicted slope')
plt.plot(true_y1, 'r', label='Y1 (Euler forward scheme)')
plt.plot(temp[:,0], 'b', label='Y1 (ML predicted)')
plt.title('Y1')
plt.legend()

plt.show()

# euler forward for y2
plt.figure()
#plt.plot(predictData2[:,1], 'b', label='Prdicted slope')
plt.plot(true_y2, 'r', label='Y2 (Euler forward scheme)')
plt.plot(temp[:,1], 'b', label='Y2 (ML predicted)')
plt.title('Y2')
plt.legend()

plt.show()


#predictData = model.predict(data_train)
## predictData  = scaler.inverse_transform(predictData )
#
"""
this is for checking if we give the whole data from t = 0 to T for different
frequency, what result we get. 
"""
omega2 = 3.0*2.0*np.pi

data2 = [[np.cos(omega2*dt*i), -omega2*np.sin(omega2*dt*i)] for i in range(nt)]
data2 = np.array(data2, dtype=np.float64)
data2 = data2

target2 = [[(data2[i,0]-data2[i-1,0])/dt, (data2[i,1]-data2[i-1,1])/dt ] for i in range(1,nt)]
target2 = np.array(target2, dtype=np.float64)
target2 = target2

# normalization
scaler2 = MinMaxScaler(feature_range=(0, 1))
data2 = scaler2.fit_transform(data2)
target2 = scaler2.fit_transform(target2)


# data for training

data_train2 = data2[1:100].reshape(99,1,2)
target_train2 = target2
predictData2 = model.predict(data_train2)

target_train2 = scaler.inverse_transform(target_train2)
predictData2 = scaler.inverse_transform(predictData2)

# slope for y1
plt.figure()
plt.plot(predictData2[:,0], 'b', label='Prdicted slope')
plt.plot(target_train2[:,0], 'r', label='Actual slope')
plt.title('Predicted and actual slope Y1')
plt.legend()

plt.show()

# slope for y2
plt.figure()
plt.plot(predictData2[:,1], 'b', label='Prdicted slope')
plt.plot(target_train2[:,1], 'r', label='Actual slope')
plt.title('Predicted and actual slope Y2')
plt.legend()

plt.show()

del model
