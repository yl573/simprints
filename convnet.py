import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

train = pickle.load(open('./data.pickle', 'rb'))
X_train = train['X_train']
X_test = train['X_test']
y_train = train['y_train']
y_test = train['y_test']

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test, y_test), shuffle=True)


