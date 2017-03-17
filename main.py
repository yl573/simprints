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

# config params
batch_size = 50
epochs = 30

train = pickle.load(open('./data.pickle', 'rb'))
X_train = train['X_train']
X_test = train['X_test']
y_train = train['y_train']
y_test = train['y_test']

model = Sequential()
model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.4))


model.add(Flatten())
model.add(Dense(32, activation='softmax'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), shuffle=True)
results = model.evaluate(X_test, y_test, batch_size=batch_size)
print(results[1])
model.save_weights("./output/weights_" + str(results[1]) + ".convnet")



file = open("test.output","w")
print("testing model")

# loading test set
test = pickle.load(open('./test.pickle', 'rb'))
X_test_raw = test["X_test_raw"]
X_test_raw = np.array(X_test_raw)
X_test_raw = np.resize(X_test_raw, (X_test_raw.shape[0], X_test_raw.shape[1], X_test_raw.shape[2], 1))   


types = ['A', 'L', 'R', 'T', 'W']
results = model.predict(X_test_raw)
#print(results)

for i in range(len(results)):
    print("Prediction for fingerprint " + str(i + 1) + " : " + types[np.argmax(results[i])])
    file.write("Prediction for fingerprint " + str(i + 1) + " : " + types[np.argmax(results[i])] + "\n")
    for j in range(len(results[i])):
        print("   Probability that it is "+ types[j] + " : " + str(results[i][j]))
        file.write("   Probability that it is "+ types[j] + " : " + str(results[i][j]))

file.close()
        