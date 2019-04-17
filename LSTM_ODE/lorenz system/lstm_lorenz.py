#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:38:12 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from scipy.integrate import odeint

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras import initializers

#--------------------------------------------------------------------------------------------------------------#
# Lorenz system
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
  x, y, z = state  # unpack the state vector
  return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

state0 = [1.508870,-1.531271, 25.46091]
t_init  = 0.0  # Initial time
t_final = 20.0 # Final time
dt = 0.05
t = np.arange(t_init, t_final, dt)
states = odeint(f, state0, t)
nsamples = int((t_final-t_init)/dt)

#--------------------------------------------------------------------------------------------------------------#
training_set = states[0:nsamples:] # training data between t_init and (t_final/2)
t = t.reshape(nsamples,1)
m,n = training_set.shape
training_set = np.hstack((t[0:m], training_set[0:m]))
lookback = 4   # history for next time step prediction 
slopenet = "LSTM"

# scale the data between (-1,1) for tanh activation function
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(-1,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled.shape
training_set = training_set_scaled

def create_training_data_lstm(training_set, m, n, dt, lookback):
    """ 
    This function creates the input and output of the neural network. The function takes the
    training data as an input and creates the input for LSTM neural network based on the lookback.
    The input for the LSTM is 3-dimensional matrix.
    For lookback = 2;  [y(n-2), y(n-1)] ------> [y(n)]
    """
    ytrain = [training_set[i+1,1:] for i in range(lookback-1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-lookback,lookback,n+1))
    for i in range(m-lookback):
        a = training_set[i]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j]))
        xtrain[i] = a
    
    return xtrain, ytrain

xtrain, ytrain = create_training_data_lstm(training_set, m, n, dt, lookback)

#--------------------------------------------------------------------------------------------------------------#
model = Sequential()

model.add(LSTM(40, input_shape=(lookback, n+1), return_sequences=True, activation='tanh', kernel_initializer='glorot_normal'))
model.add(LSTM(40, input_shape=(lookback, n+1), activation='tanh', kernel_initializer='glorot_normal'))
model.add(Dense(n))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # compile the model

history = model.fit(xtrain, ytrain, nb_epoch=400, batch_size=50, validation_split=0.2) # run the model

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------------------#
testing_set = states
testing_set = np.hstack((t[0:m], testing_set[0:m]))
# scales the data for testing
testing_set_scaled = sc.fit_transform(testing_set)
testing_set_scaled.shape
testing_set= testing_set_scaled

m,n = testing_set.shape
ytest = np.zeros((1,lookback,n))
ytest_ml = np.zeros((m,n-1))


for i in range(lookback):
    # create data for testing at first time step
    ytest[0,i,:] = testing_set[i]
    ytest_ml[i] = testing_set[i,1:]

for i in range(lookback,m):
    slope_ml = model.predict(ytest) # slope from LSTM/ ML model
    ytest_ml[i] = slope_ml # assign variable at next time ste y(n)
    e = ytest   # temporaty variable
    for j in range(lookback-1):
        e[0,j,:] = e[0,j+1,:]
    t_temp = training_set[i,0]
    t_temp = t_temp.reshape(1,1)
    slope_ml = np.hstack((t_temp, slope_ml))
    e[0,lookback-1,:] = slope_ml  # add the solution predicted y(n)
    ytest = e # update the input for the variable prediction at time step (n+1)
    

#scale the data back to original scale
t_ml = training_set[:,0]
t_ml = t_ml.reshape((m,1))
ytest_ml = np.hstack((t_ml, ytest_ml[0:m]))
ytest_ml_unscaled = sc.inverse_transform(ytest_ml)
ytest_ml_unscaled.shape
ytest_ml= ytest_ml_unscaled
ytest_ml = ytest_ml[:,1:]

testing_set = testing_set[:,0:]
testing_set_unscaled = sc.inverse_transform(testing_set)
testing_set_unscaled.shape
testing_set= testing_set_unscaled
testing_set = testing_set[:,1:]

#--------------------------------------------------------------------------------------------------------------#
# export the data in .csv file
filename = slopenet+'_p=0'+str(lookback)+'.csv'
t = t.reshape(m,1)
results = np.hstack((t, testing_set, ytest_ml))
np.savetxt(filename, results, delimiter=",")
    
def plot_results_lorenz(dt1, slopenet, lookback):
    """ Plotting script 
    blue color is for training insample data and green is for outsample sata 
    """
    filename = slopenet+'_p=0'+str(lookback)+'.csv'
    solution = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
    m,n = solution.shape
    time = solution[:,0]
    ytrue = solution[:,1:int((n-1)/2+1)]
    ytestplot = solution[0:int(m/2),int((n-1)/2+1):]
    ypredplot = solution[int(m/2):,int((n-1)/2+1):]
    for i in range(int(n/2)):
        plt.figure()    
        plt.plot(time,ytrue[:,i], 'r-', label=r'$y_'+str(i+1)+'$'+' (True)')
        plt.plot(time[0:int(m/2)],ytestplot[:,i], 'b-', label=r'$y_'+str(i+1)+'$'+' (ML Test)')
        plt.plot(time[int(m/2):],ypredplot[:,i], 'g-', label=r'$y_'+str(1+1)+'$'+' (ML Pred)')
        plt.ylabel('Response')
        plt.xlabel('Time')
        plt.legend(loc='best')
        plt.show()
        
plot_results_lorenz(dt, slopenet, lookback) # plot ML prediction and true data



