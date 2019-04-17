#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:19:23 2019

@author: user1
"""
legs = 1 
slopenet = "EULER" 
problem = "ODE"

import os
if os.path.isfile('best_model.hd5'):
    os.remove('best_model.hd5')

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

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
dt = 0.01
t = np.arange(t_init, t_final, dt)
nsamples = int((t_final-t_init)/dt)
states = odeint(f, state0, t)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:,0], states[:,1], states[:,2])
plt.show()

#--------------------------------------------------------------------------------------------------------------#
training_set = states[0:nsamples,:]
m,n = training_set.shape
n = n+1

def create_training_data(training_set, m, n, dt, t, nsamples):
    """ 
    This function creates the input and output of the neural network. The function takes the
    training data as an input and creates the input for DNN.
    The input for the LSTM is 3-dimensional matrix.
    For lookback = 2;  [y(n)] ------> [(y(n+1)-y(n))/dt]
    """
    ytrain = [(training_set[i+1]-training_set[i])/dt for i in range(m-1)]
    ytrain = np.array(ytrain)
    t = t.reshape(nsamples,1)
    xtrain = np.hstack((t[0:m-1], training_set[0:m-1]))
    return xtrain, ytrain

xtrain, ytrain = create_training_data(training_set, m, n, dt, t, nsamples)

# scale the iput data between (-1,1) for tanh activation function
from sklearn.preprocessing import MinMaxScaler
sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(xtrain)
xtrain_scaled = sc_input.transform(xtrain)
xtrain_scaled.shape
xtrain = xtrain_scaled

# scale the output data between (-1,1) for tanh activation function
from sklearn.preprocessing import MinMaxScaler
sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(ytrain)
ytrain_scaled = sc_output.transform(ytrain)
ytrain_scaled.shape
ytrain = ytrain_scaled

#--------------------------------------------------------------------------------------------------------------#
model = Sequential()

input_layer = Input(shape=(legs*(n-1)+1,)) # input layer

# Hidden layers
x = Dense(40, activation='tanh', use_bias=True)(input_layer)
x = Dense(40, activation='tanh', use_bias=True)(x)
x = Dense(40, activation='tanh', use_bias=True)(x)
x = Dense(40, activation='tanh', use_bias=True)(x)

op_val = Dense(3, activation='linear', use_bias=True)(x) # output layer

custom_model = Model(inputs=input_layer, outputs=op_val)
filepath = "best_model.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
history_callback = custom_model.fit(xtrain, ytrain, epochs=300, batch_size=40, verbose=1, validation_split= 0.2,
                                    callbacks=callbacks_list)

loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]
scores = custom_model.evaluate(xtrain, ytrain)
print("\n%s: %.2f%%" % (custom_model.metrics_names[1], scores[1]*100))

epochs = range(1, len(loss_history) + 1)

plt.figure()
plt.semilogy(epochs, loss_history, 'b', label='Training loss')
plt.semilogy(epochs, val_loss_history, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------------------#
testing_set = states
m,n=testing_set.shape
time = t
t_temp = time[0]

#def model_predict(testing_set, m, n, dt, sc_input, sc_output, t_temp):
custom_model = load_model('best_model.hd5',custom_objects={'coeff_determination': coeff_determination})

ytest = [testing_set[0]] # start ytest = y(0)
ytest = np.array(ytest)
t_temp = t_temp.reshape(1,1)
ytest = np.hstack((t_temp,ytest))

ytest_ml = [testing_set[0]] # y_ml(0) = y_exact(0)
ytest_ml = np.array(ytest_ml)

for i in range(1,m):
    ytest_sc = sc_input.transform(ytest) # scale the input to the model
    slope_ml = custom_model.predict(ytest_sc) # predict slope from the model 
    slope_ml_sc = sc_output.inverse_transform(slope_ml) # scale the calculated slope to the training data scale
    a = ytest_ml[i-1] + 1.0*dt*slope_ml_sc[0] # calculate the cariable at next time step
    ytest_ml = np.vstack((ytest_ml, a))
    ytest = a.reshape(1,n) # update the input for next time step 
    t_temp = t_temp + dt
    ytest = np.hstack((t_temp,ytest))

#return ytest_ml

#ytest_ml = model_predict(testing_set, m, n, dt, sc_input, sc_output, t_temp)

filename = slopenet+'_p=0'+str(legs)+'.csv'
time = time.reshape(m,1)
results = np.hstack((time, testing_set, ytest_ml))
np.savetxt(filename, results, delimiter=",")
    
def plot_results_lorenz(dt1, slopenet, legs):
    filename = slopenet+'_p=0'+str(legs)+'.csv'
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
        
plot_results_lorenz(dt,slopenet, legs)