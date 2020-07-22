#!/usr/bin/env python
# coding: utf-8

# In[31]:


#Importing necessary packages
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPool2D, Input, Softmax, BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from keras .utils.np_utils import to_categorical
import pandas as pd


# In[32]:


#function to load data from the location
def loadData():
    X_train = list()
    y_train = list()
    X_test = list()
    y_test = list()
    
    data_folder = 'Images/'
    
    # print(os.listdir(data_folder))
    training_folder = 'TRAIN/'
    testing_folder = 'TEST/'
    
    # Generating the training data
    for folder in os.listdir(data_folder+training_folder):
        count = 0
        if folder == 'EOSINOPHIL':
            label = 0
        elif folder == 'LYMPHOCYTE':
            label = 1
        elif folder == 'MONOCYTE':
            label = 2
        elif folder == 'NEUTROPHIL':
            label = 3
        for image_name in os.listdir(data_folder+training_folder+folder):
            image = cv2.imread(data_folder+training_folder+folder+"/"+image_name)
#             new = img_to_array(load_image())
            if image is not None:
                new = cv2.resize(image, (80,60))
                new = np.asarray(new)
                X_train.append(new)
                y_train.append(label)
                
    # Generating the testing set
    for folder in os.listdir(data_folder+testing_folder):
        count = 0
        if folder == 'EOSINOPHIL':
            label = 0
        elif folder == 'LYMPHOCYTE':
            label = 1
        elif folder == 'MONOCYTE':
            label = 2
        elif folder == 'NEUTROPHIL':
            label = 3
        for image_name in os.listdir(data_folder+testing_folder+folder):
            image = cv2.imread(data_folder+testing_folder+folder+"/"+image_name)
#             new = img_to_array(load_image())
            if image is not None:
                new = cv2.resize(image, (80,60))
                new = np.asarray(new)
                X_test.append(new)
                y_test.append(label)
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_train), np.asarray(y_train)
#             print(type(new))
#                 print(new.shape)
#             count = count + 1
#         print("{} -> {}".format(folder, count))


# In[33]:


def Model():
    model = Sequential()
    model.add(Conv2D(32, (7,7), padding='same', activation='relu', input_shape=(60,80,3)))
    model.add(MaxPool2D(pool_size=(2,2)))
              
    model.add(Conv2D(32, (7,7), padding='valid', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
              
    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(Conv2D(64, (5,5), padding='valid', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
              
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='valid', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
              
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary)
    return model


# In[ ]:


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = loadData()
    y_train = to_categorical(y_train, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)
#     print(X_train.shape)
    model = Model()
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=0)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(accuracy)


# In[ ]:




