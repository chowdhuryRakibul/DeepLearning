#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:44:41 2020

@author: Rakibul Islam Chowdhury
"""

import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import BatchNormalization
import pickle 
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# load train and test dataset
def load_dataset():
	# load dataset
    dataset = unpickle('data_batch_1')
    X = dataset[b'data']
    y = dataset[b'labels']
    
    for i in range(2,6):
        fileName = 'data_batch_'+str(i)
        dataset = unpickle(fileName)
        X = np.append(X,np.array(dataset[b'data']),axis = 0)
        y = np.append(y,np.array(dataset[b'labels']),axis = 0)
        
    X =  X/255.0
    y = to_categorical(y)
    
    n = X.shape[0]
    X = X.reshape(n,32,32,3)
    return X,y
	
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
    plt.figure()
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
    plt.figure()
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')


X,y = load_dataset()

#split into train and test set
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,y, test_size = 0.2)
# define model
model = define_model()

# create data generator
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# prepare iterator
it_train = datagen.flow(trainX, trainY, batch_size=64)

# fit model
history = model.fit_generator(it_train, epochs=400, validation_data=(testX, testY), verbose=1)
# evaluate model
_, acc = model.evaluate(testX, testY, verbose=0)
print('> %.3f' % (acc * 100.0))

# learning curves
summarize_diagnostics(history)

y_pred = model.predict(testX)

from sklearn.metrics import confusion_matrix
#convert the probabilities into binary
row_maxes = y_pred.max(axis=1).reshape(-1, 1)
y_pred[:] = np.where(y_pred == row_maxes, 1, 0)
#create mulit label confusion matrix
cm = confusion_matrix(testY.argmax(axis=1), y_pred.argmax(axis=1))

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def accuracy(confusion_matrix):
    return np.trace(cm) / np.sum(cm)

def f1Score(label,confusion_matrix):
    ps = precision(label,confusion_matrix)
    rc = recall(label,confusion_matrix)
    return 2*(ps*rc)/(ps+rc)

print('Precision = %.3f' % precision(1,cm))
print('Recall = %.3f' % recall(1,cm))
print('Accuracy = %.3f' % accuracy(cm))
print('F1 Score = %.3f' % f1Score(1,cm))