#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:38:12 2019

@author: Suraj Pawar
"""
# import packages
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler # needed if we want to normalize input and output data. Not normalized in this code

from scipy.integrate import odeint
import matplotlib.pyplot as plt
from random import uniform

# function that returns dy/dt
def odemodel(y,t):
    dy0dt =  y[0]   * y[2]
    dy1dt = -y[1]   * y[2]
    dy2dt = -y[0]**2 + y[1]**2
    dydt  = [dy0dt, dy1dt, dy2dt]
    return dydt

# time points
t_init  = 0.0  # Initial time
t_final = 10.0 # Final time
nt_steps = 200 # Number of time steps
t = np.linspace(0,t_final, num=nt_steps)
dt = (t_final - t_init)/nt_steps
nsamples = 250

# initial condition
y0 = [1, 0.1, 0]

# solve ODE
y = odeint(odemodel,y0,t)
# Note more sophisticated ode integrators should be used

# plot results for input data
plt.plot(t, y[:,0], 'r-', label=r'$y_1$') 
plt.plot(t, y[:,1], 'g-', label=r'$y_2$') 
plt.plot(t, y[:,2], 'b-', label=r'$y_3$') 
plt.ylabel('response')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()

# calculate slope
yout = [(y[i]-y[i-1])/dt for i in range(1,len(y))]
yout.insert(0, [0, 0, 0]) # slope at time t = 0
yout = np.array(yout)

# training data always contatins ode data with initial condition [1,0.1,0] as calculated above
xtrain = y
# for LSTM the input has to be 3-dimensional array with the dimension [samples, time. features]
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/?unapproved=474658&moderation-hash=87553a7d66026ce022753be31629fd1c#comment-474658
# output shape is often [samples, features]
xtrain = xtrain.reshape(nt_steps,1,3)
ytrain = yout

# additional data for training with random initial condition
for i in range(nsamples):
    # initial condition
    y2s = 0.1*uniform(-1,1)
    y0 = [1.0, y2s, 0.0]
    # solve ode
    y = odeint(odemodel, y0, t)
    # calculate slope
    yout = [(y[j]-y[j-1])/dt for j in range(1,len(y))]
    yout.insert(0, [0, 0, 0]) # slope at time t = 0
    yout = np.array(yout)
    # reshape input and add in the previous input train data
    y = y.reshape(nt_steps,1,3)
    xtrain = np.vstack((xtrain, y))
    ytrain = np.vstack((ytrain, yout))

    
# create the LSTM model
model = Sequential()
#model.add(LSTM(3, input_shape=(1, 3), return_sequences=True, activation='tanh'))
#model.add(LSTM(6, input_shape=(1, 3), return_sequences=True, activation='tanh'))
model.add(LSTM(6, input_shape=(1, 3), return_sequences=True, activation='tanh'))
model.add(LSTM(6, input_shape=(1, 3), activation='tanh'))
model.add(Dense(3))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(xtrain, ytrain, nb_epoch=30, batch_size=100, validation_split=0.3)

# training and validation loss. Plot loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# create input at t= 0 for the model testing
ytest = [1.0, 0.05, 0.0]
ytest = np.array(ytest)
ytest = ytest.reshape(1,1,3)

# create an array to store ml predicted y
ytest_ml = [1.0, 0.05, 0.0]
ytest_ml= np.array(ytest_ml)
ytest_ml = ytest_ml.reshape(1,3)

for i in range(1,nt_steps):
    slope_ml = model.predict(ytest) # slope from LSTM/ ML model
    a = ytest_ml[i-1,0] + slope_ml[i-1,0]*dt # y1 at next time step
    b = ytest_ml[i-1,1] + slope_ml[i-1,1]*dt # y2 at next time step
    c = ytest_ml[i-1,2] + slope_ml[i-1,2]*dt # y3 at next time step
    d = [a,b,c] # [y1, y2, y3] at (n+1)th step
    d = np.array(d)
    ytest_ml = np.vstack((ytest_ml, d))
    d = d.reshape(1,1,3) # create 3D array to be added in test data
    ytest = np.vstack((ytest, d)) # add [y1, y2, y3] at (n+1) to input test for next slope prediction

# initial condition
y0test = [1, 0.05, 0]

# solve ODE
ytode = odeint(odemodel,y0test,t)

plt.figure()    
plt.plot(t, ytest_ml[:,0], 'c-', label=r'$y_1 ML$') 
plt.plot(t, ytest_ml[:,1], 'm-', label=r'$y_2 ML$') 
plt.plot(t, ytest_ml[:,2], 'y-', label=r'$y_3 ML$') 

plt.plot(t, ytode[:,0], 'r-', label=r'$y_1$') 
plt.plot(t, ytode[:,1], 'g-', label=r'$y_2$') 
plt.plot(t, ytode[:,2], 'b-', label=r'$y_3$') 

plt.ylabel('response ML')
plt.xlabel('time ML')
plt.legend(loc='best')
plt.show()