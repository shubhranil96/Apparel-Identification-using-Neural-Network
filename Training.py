from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from keras import callbacks
import time

def train():
    train = pd.read_csv('C:\\Users\\Shreya\\Final project\\train\\train.csv')

    #train.head()

    #train.shape

    train_image = []
    for i in tqdm(range(train.shape[0])):
        img = image.load_img('C:/Users/Shreya/Final project/train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,1), color_mode="grayscale")
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
    X = np.array(train_image)
    #X.shape

    y=train['label'].values
    y = to_categorical(y)

    #y.shape

    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    target_dir = './models/'
    if not os.path.exists(target_dir):
      os.mkdir(target_dir)
    model.save('./models/model.h5')
    model.save_weights('./models/weights.h5')