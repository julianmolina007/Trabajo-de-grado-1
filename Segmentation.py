# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 02:55:38 2021

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

def segment_cough(x,fs, cough_padding=0.2,min_cough_len=0.2, th_l_multiplier = 0.1, th_h_multiplier = 2):
                
    cough_mask = np.array([False]*len(x))
    

    #Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h =  th_h_multiplier*rms

    #Segment coughs
    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0
    
    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding>min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end=i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding>min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
            else:
                below_th_counter = 0
        else:
            if sample>seg_th_h:
                cough_start = i-padding if (i-padding >=0) else 0
                cough_in_progress = True
    
    return coughSegments, cough_mask

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder1.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder1-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder1 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
print(names_audio[1:10])
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder1.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder2.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder2-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder2 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder2.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder3.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder3-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder3 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder3.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder4.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder4-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder4 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder4.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder5.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder5-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder5 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder5.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())
              
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder6.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder6-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder6 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder6.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder7.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder7-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder7 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder7.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder8.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder8-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder8 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder8.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder9.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder9-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder9 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder9.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

header = 'slice_file_name classID'
header = header.split()
#Escritura del archivo de metadatos folder1
file_1 = open('metadatos tos folder10.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header) 

df = pd.read_csv ('C:/Users/Usuario/Documents/Python/metadata_compiled.csv')
path = 'C:/Users/Usuario/Documents/content/Status/Folder10-cough/'
newpath = 'C:/Users/Usuario/Documents/content/Status/Folder10 tos datos aumentados/'
aux = 0
names_audio = df.uuid.to_numpy()
status = df.status.to_numpy()

for count, names in enumerate(names_audio):
  if os.path.isfile(path + names + '.wav'):

      x,fs = librosa.load (path + names + '.wav',sr = None)
      segments_cough,mascara = segment_cough(x,fs)

      for i in range(0,len(segments_cough)):

          sf.write(newpath + names + "Cough_segment" + str(i) + '.wav',segments_cough[i], fs, 'PCM_16')

          time_stretch1 = time_stretch (segments_cough[i],0.94)
          sf.write(newpath + names+ "Cough_segment_" + str(i) +'time_stretch1.wav' ,time_stretch1, fs, 'PCM_16')

          time_stretch2 = time_stretch (segments_cough[i],0.8)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch2.wav' ,time_stretch2, fs, 'PCM_16')
          
          time_stretch3 = time_stretch (segments_cough[i],1.06)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch3.wav' ,time_stretch3, fs, 'PCM_16')
          
          time_stretch4 = time_stretch (segments_cough[i],1.24)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'time_stretch4.wav' ,time_stretch4, fs, 'PCM_16')

          pitch_shift1 = pitch_shift(segments_cough[i],fs,-2)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift1.wav',pitch_shift1, fs, 'PCM_16') 

          pitch_shift2 = pitch_shift(segments_cough[i],fs,2)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift2.wav' ,pitch_shift2, fs, 'PCM_16')
          
          pitch_shift3 = pitch_shift(segments_cough[i],fs,-3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) + 'pitch_shift3.wav',pitch_shift3, fs, 'PCM_16') 

          pitch_shift4 = pitch_shift(segments_cough[i],fs,3.5)
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'pitch_shift4.wav' ,pitch_shift4, fs, 'PCM_16')

          wav_n = segments_cough[i] + 0.009*np.random.normal(0,1,len(segments_cough[i]))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_noise.wav',wav_n, fs, 'PCM_16')

          wav_roll = np.roll(segments_cough[i],int(fs/10))
          sf.write(newpath + names + "Cough_segment_" + str(i) +  'wav_roll.wav',wav_roll, fs, 'PCM_16')


          name_first = names + "Cough_segment_" + str(i) + '.wav'
          
          name_time_stretch = names + "Cough_segment_" + str(i) + 'time_stretch1.wav'
          name_time_stretch2 = names + "Cough_segment_" + str(i) + 'time_stretch2.wav'
          name_time_stretch3 = names + "Cough_segment_" + str(i) + 'time_stretch3.wav'
          name_time_stretch4 = names + "Cough_segment_" + str(i) + 'time_stretch4.wav'
          
          name_pitch_shift1 = names + "Cough_segment_" + str(i) + 'pitch_shift1.wav'
          name_pitch_shift2 = names + "Cough_segment_" + str(i) +  'pitch_shift2.wav'
          name_pitch_shift3 = names + "Cough_segment_" + str(i) + 'pitch_shift3.wav'
          name_pitch_shift4 = names + "Cough_segment_" + str(i) +  'pitch_shift4.wav'
          
          name_wav_n = names + "Cough_segment_" + str(i) +  'wav_noise.wav'
          
          wav_roll = names + "Cough_segment_" + str(i) +  'wav_roll.wav'

          if status[count]=='healthy':
              aux = 1
          elif status[count]=='COVID-19': 
              aux = 0
        

          to_append_0 = f'{name_first} {aux}'
          
          to_append_1 = f'{name_time_stretch } {aux}' 
          to_append_2 = f'{name_time_stretch2} {aux}'
          to_append_3 = f'{name_time_stretch3} {aux}' 
          to_append_4 = f'{name_time_stretch4} {aux}' 
          
          to_append_5 = f'{name_pitch_shift1} {aux}'
          to_append_6 = f'{name_pitch_shift2} {aux}'
          to_append_7 = f'{name_pitch_shift3} {aux}'
          to_append_8 = f'{name_pitch_shift4} {aux}'
          
          to_append_9 = f'{name_wav_n} {aux}'
          to_append_10 = f'{wav_roll} {aux}'

          file_1 = open('metadatos tos folder10.csv', 'a', newline='')
          with file_1:
              writer = csv.writer(file_1)
              writer.writerow(to_append_0.split())
              writer.writerow(to_append_1.split())
              writer.writerow(to_append_2.split())
              writer.writerow(to_append_3.split())
              writer.writerow(to_append_4.split())
              writer.writerow(to_append_5.split())
              writer.writerow(to_append_6.split())
              writer.writerow(to_append_7.split())
              writer.writerow(to_append_8.split())
              writer.writerow(to_append_9.split())
              writer.writerow(to_append_10.split())

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################



