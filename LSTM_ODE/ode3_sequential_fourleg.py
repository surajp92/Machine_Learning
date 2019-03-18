#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:19:23 2019

@author: user1
"""
# import packages
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

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
rsamples = 1000

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
yout = [y[i+4] for i in range(len(y)-4)]
# yout.insert(len(y)-1, y[len(y)-1]) # solution at y(tn) = y(t_n-1)
yout = np.array(yout)

# training data always contatins ode data with initial condition [1,0.1,0] as calculated above
# temporaty traininig data
xt = y
# for LSTM the input has to be 3-dimensional array with the dimension [samples, time. features]
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/?unapproved=474658&moderation-hash=87553a7d66026ce022753be31629fd1c#comment-474658
# output shape is often [samples, features]
ytrain = yout

xtrain = np.zeros((len(y)-4,12))

# data for four leg [y(n-3), y(n-2), y(n-1), y(n)]
for i in range(3,len(y)-1):
    aa = [y[i-3,:], y[i-2,:], y[i-1,:], y[i,:]]
    aa = np.array(aa)
    aa = aa.reshape(1,12)
    xtrain[i-3] = aa
    
# additional data for training with random initial condition
for i in range(nsamples):
    # initial condition
    y2s = 0.1*uniform(-1,1)
    y0 = [1.0, y2s, 0.0]
    # solve ode
    y = odeint(odemodel, y0, t)
    # calculate slope
    yout = [y[j+4] for j in range(len(y)-4)]
    # yout.insert(len(y)-1, y[len(y)-1]) # solution at y(tn) = y(t_n-1)
    yout = np.array(yout)
    ytrain = np.vstack((ytrain, yout))
    # temporay storage
    xtemp = np.zeros((len(y)-4,12))
    
    for k in range(3,len(y)-1):
        aa = [y[k-3,:], y[k-2,:], y[k-1,:], y[k,:]]
        aa = np.array(aa)
        aa = aa.reshape(1,12)
        xtemp[k-3] = aa
    
    xtrain = np.vstack((xtrain, xtemp))
    
# randomly sample data
indices = np.random.randint(0,xtrain.shape[0],rsamples)

xtrain = xtrain[indices]
ytrain = ytrain[indices]

xtrain = xtrain.reshape(rsamples,1,12)

# create the LSTM model
model = Sequential()
#model.add(LSTM(3, input_shape=(1, 3), return_sequences=True, activation='tanh'))
#model.add(LSTM(120, input_shape=(1, 12), return_sequences=True, activation='tanh'))
model.add(LSTM(120, input_shape=(1, 12), return_sequences=True, activation='tanh'))
model.add(LSTM(120, input_shape=(1, 12), activation='tanh'))
model.add(Dense(3))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# run the model
history = model.fit(xtrain, ytrain, nb_epoch=800, batch_size=100, validation_split=0.3)

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

# initial condition
y0test = [1, 0.1, 0]

# solve ODE
ytode = odeint(odemodel,y0test,t)

# create input at t= 0 for the model testing
ytest = [ytode[0], ytode[1], ytode[2], ytode[3] ]
ytest = np.array(ytest)
ytest = ytest.reshape(1,1,12)

# create an array to store ml predicted y
ytest_ml = np.zeros((nt_steps,3))
# ytest_ml = ode solution for first four time steps
ytest_ml[0] = ytode[0]
ytest_ml[1] = ytode[1]
ytest_ml[2] = ytode[2]
ytest_ml[3] = ytode[3]

for i in range(4,nt_steps):
    slope_ml = model.predict(ytest) # slope from LSTM/ ML model
    a = slope_ml[i-4,0] # y1 at next time step
    b = slope_ml[i-4,1] # y2 at next time step
    c = slope_ml[i-4,2] # y3 at next time step
    d = [a,b,c] # [y1, y2, y3] at (n+1)th step
    d = np.array(d)
    ytest_ml[i] = d
    d = d.reshape(1,3)
    ytemp = ytest[i-4]
    e = np.concatenate((ytemp,d), axis = 1)
    ee = e[0,3:15]
    ee = ee.reshape(1,1,12)
    ytest = np.vstack((ytest, ee)) # add [y1, y2, y3] at (n+1) to input test for next slope prediction

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