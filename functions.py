# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 20:03:48 2020

@author: ritug
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
#from contextlib import contextmanager
import crepe

#show entire Numpy array
"""doptions = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    try:
        yield
    finally:
        np.set_printoptions(**oldoptions)
"""

#LIBROSA load file with sr. Return numpy array with zeros trimmed
def load(path, sr=40000):
    y, sr = librosa.load(path, sr, duration=12) # restricting it to twelve seconds because numpy arrays have to be equal for combine()
   # trimmed_audio = np.trim_zeros(y)    
    return y

def find_tempo(audio_array, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=audio_array, sr=sr)
    bt_size = (1000/(tempo/60))
    return tempo, bt_size
    

#given audio array return numpy array of fundamental frequencies
def find_freq(audio_array, sr, min_note_length):
    tempo, bt_size = find_tempo(audio_array, sr)
    stp_size = bt_size*min_note_length
    time, frequency, confidence, activation = crepe.predict(audio_array, sr, viterbi=True, step_size=stp_size)
    return frequency
    
#given MIDI return pitches
def midi_pitches(path):
    None

# range from C3 t o
def freq_to_pitch(freq_array):
    pitches = np.around((np.log2(freq_array/440)*12)+69)
    #12*log2(fm/440 Hz) + 69 Frequency to MIDI formula
    return pitches
#http://peabody.sapp.org/class/st2/lab/notehz/

#Sum two waves - basically get two arrays and then sum them.
def combine(x, y):
    return x + y

def process_track(track, sr=40000, size=128):
    return librosa.power_to_db(librosa.feature.melspectrogram(y=track, sr=sr, n_mels=size), ref=np.max)

def mel_plotter(data):
    plt.figure(4)
    librosa.display.specshow(data, x_axis='time', y_axis='mel', sr=40000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency Spectogram')
    plt.show()
#What type of array does find_freq return?
#y = find_freq(load("C:\\Guitar\\Twinkle, Twinkle, Little Star.wav"), 40000, 1)
#print(y)
