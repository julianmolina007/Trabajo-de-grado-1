# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 02:39:29 2021

@author: Usuario
"""

import numpy as np
import pandas as pd
import os
import soundfile as sf
import struct
import sklearn
import subprocess
import IPython.display as ipd
from pathlib import Path
from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt
import librosa
import librosa.display as librosa_display
import sys
import csv
from tensorflow.keras.utils import to_categorical
from matplotlib import cm
from sklearn import preprocessing
import seaborn as sns
from tensorflow.keras import backend as K
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeavePOut
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import pickle
from sklearn import decomposition
from scipy.stats import kurtosis
from scipy.integrate import simps
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler


def trunc_signal_urban(x,fs):
    rms = np.sqrt(np.mean(np.square(x)))
    #Anterior 0.001
    #0.1
    umbral_high =  0.1*rms
    senal_recortada = []
    cough_start = 0
    cough_mask = np.array([False]*len(x))
    count_aux = 0
    cough_in_progress = False
    activacion_aux = True

    for counter, muestra in enumerate(x**2):
      if cough_in_progress:
        if count_aux < (cough_start + fs):
          count_aux = count_aux + 1
          senal_recortada.append(x[count_aux])
          
          activacion_aux = False
      
      else:
        if muestra > umbral_high and activacion_aux == True:
          cough_in_progress = True
          if (counter>=0):
            cough_start = counter
            
            count_aux = cough_start
          else:
            cough_start = 0

    salida = np.array(senal_recortada)
    cough_mask[cough_start:count_aux] = True
    return salida

HOP_LENGTH = 512        # number of samples between successive frames
WINDOW_LENGTH = 512     # length of the window in samples
N_MEL = 128            # number of Mel bands to generate

def compute_melspectrogram_with_fixed_length(audio, sampling_rate, num_of_samples=128):
    try:
        # compute a mel-scaled spectrogram
        melspectrogram = librosa.feature.melspectrogram(y=audio, 
                                                        sr=sampling_rate, 
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_LENGTH, 
                                                        n_mels=N_MEL)

        # convert a power spectrogram to decibel units (log-mel spectrogram)
        melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
        
        melspectrogram_length = melspectrogram_db.shape[1]
     
        # pad or fix the length of spectrogram 
        if melspectrogram_length != num_of_samples:
            melspectrogram_db = librosa.util.fix_length(melspectrogram_db, 
                                                        size=num_of_samples, 
                                                        axis=1, 
                                                        constant_values=(0, -80.0))
            mfcc_delta2 = librosa.feature.delta(melspectrogram_db, order=1)
            melspectrogram_db = np.dstack((melspectrogram_db,mfcc_delta2))
    except Exception as e:
        print("\nError encountered while parsing files\n>>", e)
        return None 
 
    return melspectrogram_db

def prepare_vectorsf1():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder1.csv')
    path = ('/content/Archivos de audio/Folder1 Datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

def prepare_vectorsf2():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder2.csv')
    path = ('/content/Archivos de audio/Folder2 datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

def prepare_vectorsf3():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder3.csv')
    path = ('/content/Archivos de audio/Folder3 Datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

def prepare_vectorsf4():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder4.csv')
    path = ('/content/Archivos de audio/Folder4 Datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

def prepare_vectorsf5():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder5.csv')
    path = ('/content/Archivos de audio/Folder5 Datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

def prepare_vectorsf6():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder6.csv')
    path = ('/content/Archivos de audio/Folder6 Datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

def prepare_vectorsf7():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder7.csv')
    path = ('/content/Archivos de audio/Folder7 Datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

def prepare_vectorsf8():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder8.csv')
    path = ('/content/Archivos de audio/Folder8 Datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)

            
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

def prepare_vectorsf9():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder9.csv')
    path = ('/content/Archivos de audio/Folder9 Datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

def prepare_vectorsf10():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('/content/metadatos_folder10.csv')
    path = ('/content/Archivos de audio/Folder10 Datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names)
            Signal = trunc_signal_urban(Signal,rate)
            if len(Signal) == 22050:
                data = compute_melspectrogram_with_fixed_length(Signal, rate)
                X.append(data[np.newaxis,...])    
                y.append(classes.index(labels[count]))
                aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)


def CNN_model(input_shape):
    
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(24, (5, 5), activation = 'relu',strides=(1, 1), input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((4,2), strides = (4,2), padding='same'))
    #model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(48, (5, 5), activation = 'relu',strides=(1, 1)))
    model.add(keras.layers.MaxPool2D((4,2), strides = (4,2), padding='same'))
    #model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(48, (5, 5), activation = 'relu',strides=(1, 1), padding='same'))
    #model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.5))

    model.add(keras.layers.Dense(64, activation = 'relu'))
    model.add(keras.layers.Dropout(rate=0.5))

    model.add(keras.layers.Dense(11, activation= 'softmax'))

    model.summary()
    
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy',
                  metrics = ['categorical_accuracy','AUC'])
    
    
    return model


X_train1,y_train1 = prepare_vectorsf1()
X_train2,y_train2 = prepare_vectorsf2()
X_train3,y_train3 = prepare_vectorsf3()
X_train4,y_train4 = prepare_vectorsf4()
X_test,y_test = prepare_vectorsf5()
X_train6,y_train6 = prepare_vectorsf6()
X_train7,y_train7 = prepare_vectorsf7()
X_train8,y_train8 = prepare_vectorsf8()
X_train9,y_train9 = prepare_vectorsf9()
X_train10,y_train10 = prepare_vectorsf10()

X_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_train6,X_train7,X_train8,X_train9,X_train10),axis=0)


y_train = np.concatenate((y_train1,y_train2,y_train3,y_train4,y_train6,y_train7,y_train8,y_train9,y_train10),axis=0)
y_train = to_categorical(y_train, num_classes=11)


y_test = to_categorical(y_test, num_classes=11)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import tensorflow as tf
input_shape = (X_train.shape[1],X_train.shape[2],2)
# Construir el modelo de CNN
CNN = CNN_model(input_shape)
    
# Entrenar el modelo
CNN.fit(X_train,y_train,epochs = 90, batch_size = 500, validation_data = (X_test,y_test))
CNN.save('Modelo_de_CNN_dos_canales_1.h5')