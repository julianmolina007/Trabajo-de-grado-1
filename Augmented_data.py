# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 02:56:39 2021

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
import sys
import csv
from sklearn import preprocessing
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy import signal
from sklearn.model_selection import LeavePOut
from scipy.io import wavfile
from scipy.signal import butter,filtfilt
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
from tqdm import tqdm

def time_stretch (signal, rate):
  # ampliar por 1.2 y 0.8
  data_augmented = librosa.effects.time_stretch(signal, rate = rate)
  return data_augmented

# Aumento de datos mediante desplazamiento de tono
def pitch_shift (signal,fs,n_steps):
  # ampliar por -2 y 2
  data_augmented = librosa.effects.pitch_shift(signal, fs, n_steps)
  return data_augmented
#sf.write(path + 'augmented_time1.wav',x, fs, 'PCM_16')



# Aumento de los datos y creacion del archivo de metadatos:

# Declaracion del encabezado del archivo de metadatos:
header = 'slice_file_name fold classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos folder 5.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

## Paths y definicion de los nombres para buscar los archivos de UrbandSound8K
df = pd.read_csv ('C:/Users/Usuario/Documents/Python/UrbanSound8K.csv')
path = 'C:/Users/Usuario/Documents/Python/fold5/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 5 datos aumentados/'
names_audio = df.slice_file_name.to_numpy()
type_class = df.classID.to_numpy()

# For para aumentar los datos y guardarlos en una nueva carpeta
for count, names in enumerate(names_audio):
  if os.path.isfile(path + names):
      
    x,fs = librosa.load (path + names, sr = None)
    sf.write(newpath + names ,x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + 'time_stretch1_' + names ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + 'time_stretch2_'+ names ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + 'pitch_shift1_'+ names ,pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + 'pitch_shift2_'+ names ,pitch_shift2, fs, 'PCM_16')
    
    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + 'wav_noise_'+ names ,wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + 'wav_roll_'+ names ,wav_roll, fs, 'PCM_16')
    

    name_first = names
    name_time_stretch = 'time_stretch1_'+ names 
    name_time_stretch2 = 'time_stretch2_'+ names
    name_pitch_shift1 = 'pitch_shift1_'+ names
    name_pitch_shift2 = 'pitch_shift2_'+ names
    name_wav_n = 'wav_noise_'+ names
    wav_roll = 'wav_roll_'+ names

    to_append_0 = f'{name_first} {5} {type_class[count]}'
    to_append_1 = f'{name_time_stretch } {5} {type_class[count]}'
    to_append_2 = f'{name_time_stretch2} {5} {type_class[count]}'
    to_append_3 = f'{name_pitch_shift1} {5} {type_class[count]}'
    to_append_4 = f'{name_pitch_shift2} {5} {type_class[count]}'
    to_append_5 = f'{name_wav_n} {5} {type_class[count]}'
    to_append_6 = f'{wav_roll} {5} {type_class[count]}'


    file_1 = open('metadatos folder 5.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

# Paths y definicion de los nombres para buscar los archivos del dataset coughvid
print('parte 1 terminada')

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/Python/fold5/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 5 datos aumentados/'
names_audio = df.uuid.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

    x,fs = librosa.load (path + names + '.wav',sr = None)
    sf.write(newpath + names + '.wav',x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + names + 'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + names + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + names + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + names +'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')

    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + names + 'wav_noise.wav',wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + names + 'wav_roll.wav',wav_roll, fs, 'PCM_16')
    

    name_first = names + '.wav'
    name_time_stretch = names + 'time_stretch1.wav'
    name_time_stretch2 = names + 'time_stretch2.wav'
    name_pitch_shift1 = names + 'pitch_shift1.wav'
    name_pitch_shift2 = names +'pitch_shift2.wav'
    name_wav_n = names + 'wav_noise.wav'
    wav_roll = names + 'wav_roll.wav'

    to_append_0 = f'{name_first} {5} {10}'
    to_append_1 = f'{name_time_stretch } {5} {10}'
    to_append_2 = f'{name_time_stretch2} {5} {10}'
    to_append_3 = f'{name_pitch_shift1} {5} {10}'
    to_append_4 = f'{name_pitch_shift2} {5} {10}'
    to_append_5 = f'{name_wav_n} {5} {10}'
    to_append_6 = f'{wav_roll} {5} {10}'
    
    file_1 = open('metadatos folder 5.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

print('parte 2 terminada')






###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

header = 'slice_file_name fold classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos folder 6.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

## Paths y definicion de los nombres para buscar los archivos de UrbandSound8K
df = pd.read_csv ('C:/Users/Usuario/Documents/Python/UrbanSound8K.csv')
path = 'C:/Users/Usuario/Documents/Python/fold6/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 6 datos aumentados/'
names_audio = df.slice_file_name.to_numpy()
type_class = df.classID.to_numpy()

# For para aumentar los datos y guardarlos en una nueva carpeta
for count, names in enumerate(names_audio):
  if os.path.isfile(path + names):
      
    x,fs = librosa.load (path + names, sr = None)
    sf.write(newpath + names ,x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + 'time_stretch1_' + names ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + 'time_stretch2_'+ names ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + 'pitch_shift1_'+ names ,pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + 'pitch_shift2_'+ names ,pitch_shift2, fs, 'PCM_16')
    
    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + 'wav_noise_'+ names ,wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + 'wav_roll_'+ names ,wav_roll, fs, 'PCM_16')
    

    name_first = names
    name_time_stretch = 'time_stretch1_'+ names 
    name_time_stretch2 = 'time_stretch2_'+ names
    name_pitch_shift1 = 'pitch_shift1_'+ names
    name_pitch_shift2 = 'pitch_shift2_'+ names
    name_wav_n = 'wav_noise_'+ names
    wav_roll = 'wav_roll_'+ names

    to_append_0 = f'{name_first} {6} {type_class[count]}'
    to_append_1 = f'{name_time_stretch } {6} {type_class[count]}'
    to_append_2 = f'{name_time_stretch2} {6} {type_class[count]}'
    to_append_3 = f'{name_pitch_shift1} {6} {type_class[count]}'
    to_append_4 = f'{name_pitch_shift2} {6} {type_class[count]}'
    to_append_5 = f'{name_wav_n} {6} {type_class[count]}'
    to_append_6 = f'{wav_roll} {6} {type_class[count]}'


    file_1 = open('metadatos folder 6.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

# Paths y definicion de los nombres para buscar los archivos del dataset coughvid
print('parte 1 terminada')

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/Python/fold6/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 6 datos aumentados/'
names_audio = df.uuid.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

    x,fs = librosa.load (path + names + '.wav',sr = None)
    sf.write(newpath + names + '.wav',x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + names + 'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + names + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + names + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + names +'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')

    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + names + 'wav_noise.wav',wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + names + 'wav_roll.wav',wav_roll, fs, 'PCM_16')
    

    name_first = names + '.wav'
    name_time_stretch = names + 'time_stretch1.wav'
    name_time_stretch2 = names + 'time_stretch2.wav'
    name_pitch_shift1 = names + 'pitch_shift1.wav'
    name_pitch_shift2 = names +'pitch_shift2.wav'
    name_wav_n = names + 'wav_noise.wav'
    wav_roll = names + 'wav_roll.wav'

    to_append_0 = f'{name_first} {6} {10}'
    to_append_1 = f'{name_time_stretch } {6} {10}'
    to_append_2 = f'{name_time_stretch2} {6} {10}'
    to_append_3 = f'{name_pitch_shift1} {6} {10}'
    to_append_4 = f'{name_pitch_shift2} {6} {10}'
    to_append_5 = f'{name_wav_n} {6} {10}'
    to_append_6 = f'{wav_roll} {6} {10}'
    
    file_1 = open('metadatos folder 6.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

print('parte 2 terminada')




###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

header = 'slice_file_name fold classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos folder 7.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

## Paths y definicion de los nombres para buscar los archivos de UrbandSound8K
df = pd.read_csv ('C:/Users/Usuario/Documents/Python/UrbanSound8K.csv')
path = 'C:/Users/Usuario/Documents/Python/fold7/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 7 datos aumentados/'
names_audio = df.slice_file_name.to_numpy()
type_class = df.classID.to_numpy()

# For para aumentar los datos y guardarlos en una nueva carpeta
for count, names in enumerate(names_audio):
  if os.path.isfile(path + names):
      
    x,fs = librosa.load (path + names, sr = None)
    sf.write(newpath + names ,x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + 'time_stretch1_' + names ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + 'time_stretch2_'+ names ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + 'pitch_shift1_'+ names ,pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + 'pitch_shift2_'+ names ,pitch_shift2, fs, 'PCM_16')
    
    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + 'wav_noise_'+ names ,wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + 'wav_roll_'+ names ,wav_roll, fs, 'PCM_16')
    

    name_first = names
    name_time_stretch = 'time_stretch1_'+ names 
    name_time_stretch2 = 'time_stretch2_'+ names
    name_pitch_shift1 = 'pitch_shift1_'+ names
    name_pitch_shift2 = 'pitch_shift2_'+ names
    name_wav_n = 'wav_noise_'+ names
    wav_roll = 'wav_roll_'+ names

    to_append_0 = f'{name_first} {7} {type_class[count]}'
    to_append_1 = f'{name_time_stretch } {7} {type_class[count]}'
    to_append_2 = f'{name_time_stretch2} {7} {type_class[count]}'
    to_append_3 = f'{name_pitch_shift1} {7} {type_class[count]}'
    to_append_4 = f'{name_pitch_shift2} {7} {type_class[count]}'
    to_append_5 = f'{name_wav_n} {7} {type_class[count]}'
    to_append_6 = f'{wav_roll} {7} {type_class[count]}'


    file_1 = open('metadatos folder 7.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

# Paths y definicion de los nombres para buscar los archivos del dataset coughvid
print('parte 1 terminada')

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/Python/fold7/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 7 datos aumentados/'
names_audio = df.uuid.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

    x,fs = librosa.load (path + names + '.wav',sr = None)
    sf.write(newpath + names + '.wav',x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + names + 'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + names + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + names + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + names +'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')

    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + names + 'wav_noise.wav',wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + names + 'wav_roll.wav',wav_roll, fs, 'PCM_16')
    

    name_first = names + '.wav'
    name_time_stretch = names + 'time_stretch1.wav'
    name_time_stretch2 = names + 'time_stretch2.wav'
    name_pitch_shift1 = names + 'pitch_shift1.wav'
    name_pitch_shift2 = names +'pitch_shift2.wav'
    name_wav_n = names + 'wav_noise.wav'
    wav_roll = names + 'wav_roll.wav'

    to_append_0 = f'{name_first} {7} {10}'
    to_append_1 = f'{name_time_stretch } {7} {10}'
    to_append_2 = f'{name_time_stretch2} {7} {10}'
    to_append_3 = f'{name_pitch_shift1} {7} {10}'
    to_append_4 = f'{name_pitch_shift2} {7} {10}'
    to_append_5 = f'{name_wav_n} {7} {10}'
    to_append_6 = f'{wav_roll} {7} {10}'
    
    file_1 = open('metadatos folder 7.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

print('parte 2 terminada')




###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################


header = 'slice_file_name fold classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos folder 8.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

## Paths y definicion de los nombres para buscar los archivos de UrbandSound8K
df = pd.read_csv ('C:/Users/Usuario/Documents/Python/UrbanSound8K.csv')
path = 'C:/Users/Usuario/Documents/Python/fold8/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 8 datos aumentados/'
names_audio = df.slice_file_name.to_numpy()
type_class = df.classID.to_numpy()

# For para aumentar los datos y guardarlos en una nueva carpeta
for count, names in enumerate(names_audio):
  if os.path.isfile(path + names):
      
    x,fs = librosa.load (path + names, sr = None)
    sf.write(newpath + names ,x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + 'time_stretch1_' + names ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + 'time_stretch2_'+ names ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + 'pitch_shift1_'+ names ,pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + 'pitch_shift2_'+ names ,pitch_shift2, fs, 'PCM_16')
    
    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + 'wav_noise_'+ names ,wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + 'wav_roll_'+ names ,wav_roll, fs, 'PCM_16')
    

    name_first = names
    name_time_stretch = 'time_stretch1_'+ names 
    name_time_stretch2 = 'time_stretch2_'+ names
    name_pitch_shift1 = 'pitch_shift1_'+ names
    name_pitch_shift2 = 'pitch_shift2_'+ names
    name_wav_n = 'wav_noise_'+ names
    wav_roll = 'wav_roll_'+ names

    to_append_0 = f'{name_first} {8} {type_class[count]}'
    to_append_1 = f'{name_time_stretch } {8} {type_class[count]}'
    to_append_2 = f'{name_time_stretch2} {8} {type_class[count]}'
    to_append_3 = f'{name_pitch_shift1} {8} {type_class[count]}'
    to_append_4 = f'{name_pitch_shift2} {8} {type_class[count]}'
    to_append_5 = f'{name_wav_n} {8} {type_class[count]}'
    to_append_6 = f'{wav_roll} {8} {type_class[count]}'


    file_1 = open('metadatos folder 8.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

# Paths y definicion de los nombres para buscar los archivos del dataset coughvid
print('parte 1 terminada')

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/Python/fold8/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 8 datos aumentados/'
names_audio = df.uuid.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

    x,fs = librosa.load (path + names + '.wav',sr = None)
    sf.write(newpath + names + '.wav',x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + names + 'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + names + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + names + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + names +'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')

    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + names + 'wav_noise.wav',wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + names + 'wav_roll.wav',wav_roll, fs, 'PCM_16')
    

    name_first = names + '.wav'
    name_time_stretch = names + 'time_stretch1.wav'
    name_time_stretch2 = names + 'time_stretch2.wav'
    name_pitch_shift1 = names + 'pitch_shift1.wav'
    name_pitch_shift2 = names +'pitch_shift2.wav'
    name_wav_n = names + 'wav_noise.wav'
    wav_roll = names + 'wav_roll.wav'

    to_append_0 = f'{name_first} {8} {10}'
    to_append_1 = f'{name_time_stretch } {8} {10}'
    to_append_2 = f'{name_time_stretch2} {8} {10}'
    to_append_3 = f'{name_pitch_shift1} {8} {10}'
    to_append_4 = f'{name_pitch_shift2} {8} {10}'
    to_append_5 = f'{name_wav_n} {8} {10}'
    to_append_6 = f'{wav_roll} {8} {10}'
    
    file_1 = open('metadatos folder 8.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

print('parte 2 terminada')




###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

header = 'slice_file_name fold classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos folder 9.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

## Paths y definicion de los nombres para buscar los archivos de UrbandSound8K
df = pd.read_csv ('C:/Users/Usuario/Documents/Python/UrbanSound8K.csv')
path = 'C:/Users/Usuario/Documents/Python/fold9/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 9 datos aumentados/'
names_audio = df.slice_file_name.to_numpy()
type_class = df.classID.to_numpy()

# For para aumentar los datos y guardarlos en una nueva carpeta
for count, names in enumerate(names_audio):
  if os.path.isfile(path + names):
      
    x,fs = librosa.load (path + names, sr = None)
    sf.write(newpath + names ,x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + 'time_stretch1_' + names ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + 'time_stretch2_'+ names ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + 'pitch_shift1_'+ names ,pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + 'pitch_shift2_'+ names ,pitch_shift2, fs, 'PCM_16')
    
    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + 'wav_noise_'+ names ,wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + 'wav_roll_'+ names ,wav_roll, fs, 'PCM_16')
    

    name_first = names
    name_time_stretch = 'time_stretch1_'+ names 
    name_time_stretch2 = 'time_stretch2_'+ names
    name_pitch_shift1 = 'pitch_shift1_'+ names
    name_pitch_shift2 = 'pitch_shift2_'+ names
    name_wav_n = 'wav_noise_'+ names
    wav_roll = 'wav_roll_'+ names

    to_append_0 = f'{name_first} {9} {type_class[count]}'
    to_append_1 = f'{name_time_stretch } {9} {type_class[count]}'
    to_append_2 = f'{name_time_stretch2} {9} {type_class[count]}'
    to_append_3 = f'{name_pitch_shift1} {9} {type_class[count]}'
    to_append_4 = f'{name_pitch_shift2} {9} {type_class[count]}'
    to_append_5 = f'{name_wav_n} {9} {type_class[count]}'
    to_append_6 = f'{wav_roll} {9} {type_class[count]}'


    file_1 = open('metadatos folder 9.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

# Paths y definicion de los nombres para buscar los archivos del dataset coughvid
print('parte 1 terminada')

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/Python/fold9/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 9 datos aumentados/'
names_audio = df.uuid.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

    x,fs = librosa.load (path + names + '.wav',sr = None)
    sf.write(newpath + names + '.wav',x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + names + 'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + names + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + names + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + names +'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')

    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + names + 'wav_noise.wav',wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + names + 'wav_roll.wav',wav_roll, fs, 'PCM_16')
    

    name_first = names + '.wav'
    name_time_stretch = names + 'time_stretch1.wav'
    name_time_stretch2 = names + 'time_stretch2.wav'
    name_pitch_shift1 = names + 'pitch_shift1.wav'
    name_pitch_shift2 = names +'pitch_shift2.wav'
    name_wav_n = names + 'wav_noise.wav'
    wav_roll = names + 'wav_roll.wav'

    to_append_0 = f'{name_first} {9} {10}'
    to_append_1 = f'{name_time_stretch } {9} {10}'
    to_append_2 = f'{name_time_stretch2} {9} {10}'
    to_append_3 = f'{name_pitch_shift1} {9} {10}'
    to_append_4 = f'{name_pitch_shift2} {9} {10}'
    to_append_5 = f'{name_wav_n} {9} {10}'
    to_append_6 = f'{wav_roll} {9} {10}'
    
    file_1 = open('metadatos folder 9.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

print('parte 2 terminada')




###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

header = 'slice_file_name fold classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos folder 10.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

## Paths y definicion de los nombres para buscar los archivos de UrbandSound8K
df = pd.read_csv ('C:/Users/Usuario/Documents/Python/UrbanSound8K.csv')
path = 'C:/Users/Usuario/Documents/Python/fold10/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 10 datos aumentados/'
names_audio = df.slice_file_name.to_numpy()
type_class = df.classID.to_numpy()

# For para aumentar los datos y guardarlos en una nueva carpeta
for count, names in enumerate(names_audio):
  if os.path.isfile(path + names):
      
    x,fs = librosa.load (path + names, sr = None)
    sf.write(newpath + names ,x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + 'time_stretch1_' + names ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + 'time_stretch2_'+ names ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + 'pitch_shift1_'+ names ,pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + 'pitch_shift2_'+ names ,pitch_shift2, fs, 'PCM_16')
    
    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + 'wav_noise_'+ names ,wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + 'wav_roll_'+ names ,wav_roll, fs, 'PCM_16')
    

    name_first = names
    name_time_stretch = 'time_stretch1_'+ names 
    name_time_stretch2 = 'time_stretch2_'+ names
    name_pitch_shift1 = 'pitch_shift1_'+ names
    name_pitch_shift2 = 'pitch_shift2_'+ names
    name_wav_n = 'wav_noise_'+ names
    wav_roll = 'wav_roll_'+ names

    to_append_0 = f'{name_first} {10} {type_class[count]}'
    to_append_1 = f'{name_time_stretch } {10} {type_class[count]}'
    to_append_2 = f'{name_time_stretch2} {10} {type_class[count]}'
    to_append_3 = f'{name_pitch_shift1} {10} {type_class[count]}'
    to_append_4 = f'{name_pitch_shift2} {10} {type_class[count]}'
    to_append_5 = f'{name_wav_n} {10} {type_class[count]}'
    to_append_6 = f'{wav_roll} {10} {type_class[count]}'


    file_1 = open('metadatos folder 10.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

# Paths y definicion de los nombres para buscar los archivos del dataset coughvid
print('parte 1 terminada')

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/Python/fold10/'
newpath = 'C:/Users/Usuario/Documents/Python/Fold 10 datos aumentados/'
names_audio = df.uuid.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

    x,fs = librosa.load (path + names + '.wav',sr = None)
    sf.write(newpath + names + '.wav',x, fs, 'PCM_16')

    time_stretch1 = time_stretch (x,1.2)
    sf.write(newpath + names + 'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

    time_stretch2 = time_stretch (x,0.8)
    sf.write(newpath + names + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')

    pitch_shift1 = pitch_shift(x,fs,-2)
    sf.write(newpath + names + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

    pitch_shift2 = pitch_shift(x,fs,2)
    sf.write(newpath + names +'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')

    wav_n = x + 0.009*np.random.normal(0,1,len(x))
    sf.write(newpath + names + 'wav_noise.wav',wav_n, fs, 'PCM_16')
    
    wav_roll = np.roll(x,int(fs/10))
    sf.write(newpath + names + 'wav_roll.wav',wav_roll, fs, 'PCM_16')
    

    name_first = names + '.wav'
    name_time_stretch = names + 'time_stretch1.wav'
    name_time_stretch2 = names + 'time_stretch2.wav'
    name_pitch_shift1 = names + 'pitch_shift1.wav'
    name_pitch_shift2 = names +'pitch_shift2.wav'
    name_wav_n = names + 'wav_noise.wav'
    wav_roll = names + 'wav_roll.wav'

    to_append_0 = f'{name_first} {10} {10}'
    to_append_1 = f'{name_time_stretch } {10} {10}'
    to_append_2 = f'{name_time_stretch2} {10} {10}'
    to_append_3 = f'{name_pitch_shift1} {10} {10}'
    to_append_4 = f'{name_pitch_shift2} {10} {10}'
    to_append_5 = f'{name_wav_n} {10} {10}'
    to_append_6 = f'{wav_roll} {10} {10}'
    
    file_1 = open('metadatos folder 10.csv', 'a', newline='')
    with file_1:
      writer = csv.writer(file_1)
      writer.writerow(to_append_0.split())
      writer.writerow(to_append_1.split())
      writer.writerow(to_append_2.split())
      writer.writerow(to_append_3.split())
      writer.writerow(to_append_4.split())
      writer.writerow(to_append_5.split())
      writer.writerow(to_append_6.split())

print('parte 2 terminada')



