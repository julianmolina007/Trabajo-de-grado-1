# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 02:58:10 2021

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:16:11 2021

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
import matplotlib.pyplot as plt
import librosa
import librosa.display as librosa_display
from sklearn.metrics import confusion_matrix
import sys
from tensorflow.keras.utils import to_categorical
import csv
from sklearn import preprocessing
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy import signal
from sklearn.model_selection import LeavePOut
from scipy.io import wavfile
from scipy.signal import butter,filtfilt
import tensorflow.keras as keras
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from scipy.stats import kurtosis
import scipy.signal as signal
from scipy.integrate import simps
import matplotlib.patches as mpatches
from scipy.signal import butter,filtfilt
from scipy.signal import cwt
from scipy.signal import hilbert
from scipy.signal import resample
from scipy.signal import decimate
from scipy.signal import spectrogram
from sklearn.metrics import confusion_matrix
from scipy.signal.windows import get_window
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix
def LSTM_model(input_shape):
    """Generates RNN-LSTM model
    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.summary()
    
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy','AUC'])

    return model

def LSTM_model_2(input_shape):
    model = keras.Sequential()

    model.add(keras.layers.LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
    model.add(keras.layers.LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(keras.layers.Dense(2, activation="softmax"))
    model.summary()

    
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='BinaryCrossentropy',
                  metrics=['accuracy','AUC'])

    return model

def preprocesamiento(x,fs):
  fs_downsample = 11025*2
  if len(x.shape)>1:
        x = np.mean(x,axis=1)   
  b, a = butter(4, fs_downsample/fs, btype='lowpass') # 4th order butter lowpass filter
  x = filtfilt(b, a, x)
  x = signal.decimate(x, int(fs/fs_downsample))
  return np.float32(x), fs_downsample


def prepare_vectorsf1():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=0
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder2.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder2 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    print(names_audio[1:10])
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1
       # print(aux2)

        if os.path.isfile(path + names):
            
            aux+=1
           # print(aux)
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)
            print(mfcc1.shape)
            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)
            print(spectral_center.shape)

            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)
            print(zero.shape)
            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)
            print(rolloff.shape)

            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
            print(spec_bw.shape)

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
            print(chromagram.shape)

            
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def prepare_vectorsf2():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder2.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder2 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)

            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)


            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)

            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)


            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
    

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
   

            spectral_contrast = librosa.feature.spectral_contrast(y=Signal, sr=rate, hop_length=512)
            spectral_contrast = librosa.util.fix_length(spectral_contrast, size=30, axis=1)
    
            
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T, spectral_contrast.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
def prepare_vectorsf3():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder3.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder3 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)
           
            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)
          

            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)
         
            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)
      

            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
         

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
           

            spectral_contrast = librosa.feature.spectral_contrast(y=Signal, sr=rate, hop_length=512)
            spectral_contrast = librosa.util.fix_length(spectral_contrast, size=30, axis=1)
          
            
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T, spectral_contrast.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def prepare_vectorsf4():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder4.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder4 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)
        
            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)
          

            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)

            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)
      

            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
          

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
           

            spectral_contrast = librosa.feature.spectral_contrast(y=Signal, sr=rate, hop_length=512)
            spectral_contrast = librosa.util.fix_length(spectral_contrast, size=30, axis=1)
      
            
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T, spectral_contrast.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def prepare_vectorsf5():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder5.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder5 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)
        
            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)
      

            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)
  
            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)
      

            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
            

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
           

            spectral_contrast = librosa.feature.spectral_contrast(y=Signal, sr=rate, hop_length=512)
            spectral_contrast = librosa.util.fix_length(spectral_contrast, size=30, axis=1)
         
            
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T, spectral_contrast.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def prepare_vectorsf6():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder6.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder6 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)
         
            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)
  

            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)
    
            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)
        

            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
          

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
         

            spectral_contrast = librosa.feature.spectral_contrast(y=Signal, sr=rate, hop_length=512)
            spectral_contrast = librosa.util.fix_length(spectral_contrast, size=30, axis=1)
            
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T, spectral_contrast.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def prepare_vectorsf7():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder7.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder7 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)
       
            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)
    

            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)
        
            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)
   

            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
          

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
          

            spectral_contrast = librosa.feature.spectral_contrast(y=Signal, sr=rate, hop_length=512)
            spectral_contrast = librosa.util.fix_length(spectral_contrast, size=30, axis=1)
        
            
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T, spectral_contrast.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def prepare_vectorsf8():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder8.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder8 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)
  
            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)
        

            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)
    
            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)
         

            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
           

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
          

            spectral_contrast = librosa.feature.spectral_contrast(y=Signal, sr=rate, hop_length=512)
            spectral_contrast = librosa.util.fix_length(spectral_contrast, size=30, axis=1)
           
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T, spectral_contrast.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def prepare_vectorsf9():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder9.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder9 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)
  
            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)
      

            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)
           
            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)
           

            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
      

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
        

            spectral_contrast = librosa.feature.spectral_contrast(y=Signal, sr=rate, hop_length=512)
            spectral_contrast = librosa.util.fix_length(spectral_contrast, size=30, axis=1)
           
            
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T, spectral_contrast.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################



def prepare_vectorsf10():
    X = []
    y = []
    ###### preparar el vector X para folder 2: ###########
    aux = 0
    aux2=1
    df = pd.read_csv('C:/Users/Usuario/Documents/Python/metadatos tos folder10.csv')
    path = ('C:/Users/Usuario/Documents/content/Status/Folder10 tos datos aumentados/')
    names_audio = df.slice_file_name.to_numpy()
    classes = list(np.unique(df.classID))
    labels = df.classID.to_numpy()
    
    for count, names in enumerate(names_audio):
        aux2+=1

        if os.path.isfile(path + names):
            
            Signal , rate = librosa.load(path + names, sr=None)
            Signal, rate = preprocesamiento(Signal, rate)
            
            mfcc1 = librosa.feature.mfcc(y=Signal, sr=rate, hop_length=512, n_mfcc=13 )
            mfcc1 = librosa.util.fix_length(mfcc1, size=30, axis=1)
         
            
            spectral_center = librosa.feature.spectral_centroid(y=Signal, sr=rate, hop_length=512)
            spectral_center = librosa.util.fix_length(spectral_center, size=30, axis=1)
            

            zero = librosa.feature.zero_crossing_rate(Signal, hop_length=512)
            zero = librosa.util.fix_length(zero, size=30, axis=1)
         
            
            rolloff = librosa.feature.spectral_rolloff(y=Signal, sr=rate, hop_length=512)
            rolloff = librosa.util.fix_length(rolloff, size=30, axis=1)
          

            spec_bw = librosa.feature.spectral_bandwidth(y=Signal, sr=rate, hop_length=512)
            spec_bw = librosa.util.fix_length(spec_bw, size=30, axis=1)
          

            chromagram = librosa.feature.chroma_stft(y=Signal, sr=rate, hop_length=512)
            chromagram = librosa.util.fix_length(chromagram, size=30, axis=1)
          

            spectral_contrast = librosa.feature.spectral_contrast(y=Signal, sr=rate, hop_length=512)
            spectral_contrast = librosa.util.fix_length(spectral_contrast, size=30, axis=1)
        
            
            data = np.concatenate((mfcc1.T, spectral_center.T, zero.T, rolloff.T, spec_bw.T, chromagram.T, spectral_contrast.T), axis=1)

            #data = mfcc1.T
            X.append(data[np.newaxis,...])    
            y.append(classes.index(labels[count]))
            aux+=1
    output = np.concatenate(X,axis=0)
    print('total de muestras en folder:', aux2)  
    print('Muestras añadidas al vector:', aux)
    return output,np.array(y)
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

X1,y1 = prepare_vectorsf1()
X2,y2 = prepare_vectorsf2()
X3,y3 = prepare_vectorsf3()
X4,y4 = prepare_vectorsf4()
X5,y5 = prepare_vectorsf5()
X6,y6 = prepare_vectorsf6()
X7,y7 = prepare_vectorsf7()
X8,y8 = prepare_vectorsf8()
X9,y9 = prepare_vectorsf9()
X10,y10 = prepare_vectorsf10()

X_train = np.concatenate((X1,X2,X3,X4,X6,X7,X8,X9,X10),axis=0)
y_train = np.concatenate((y1,y2,y3,y4,y6,y7,y8,y9,y10),axis=0)
y_train = to_categorical(y_train, num_classes=2)

X_test = X5
y_test = to_categorical(y5, num_classes=2)


input_shape = (X_train.shape[1], X_train.shape[2])
# Construir el modelo de CNN
LSTM1 = LSTM_model(input_shape)


# Entrenar el modelo
LSTM1.fit(X_train,y_train,epochs = 90, batch_size = 500, validation_data = (X_test,y_test))

def predict (model_LSTM,X):
    prediction = model_LSTM.predict(X)
    return prediction

y_pred = predict(LSTM1,X_test)
cm = confusion_matrix(y_target = y_test.argmax(axis=1),y_predicted = y_pred.argmax(axis=1), binary=True )
fig,ax = plot_confusion_matrix(conf_mat=cm)
plt.show()
# LSTM.fit(X_train,y_train,epochs = 100, batch_size = 500, validation_data = (X_test,y_test))
# LSTM.fit(X_train,y_train,epochs = 50, batch_size = 500, validation_data = (X_test,y_test))
