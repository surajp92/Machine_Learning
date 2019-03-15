#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:34:43 2019

@author: user1
"""
print('Newswire Classification')

# from hack import hack

# hack()

from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)


print('train data: '+ str(len(train_data)))
print('test data: '+ str(len(test_data)))

print(train_data[10])

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# create a one-hot (categorical) encoding
def to_one_hot(labels, dimension= 46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)


from keras import models
from keras import layers

model = models.Sequential()

model.add(layers.Dense(46, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(46, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# actual training
history = model.fit(partial_x_train,    # data input
                    partial_y_train,    # labels-correct answer (supervised learning)
                    epochs=20,          # maximum iteration
                    batch_size=516,     # backpropogation
                    validation_data=(x_val, y_val)) # not necessary, 

import matplotlib.pyplot as plt
 
loss = history.history['loss']
val_loss = history['val_loss']



        