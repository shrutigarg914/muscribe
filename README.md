# MuScribe

A tool to predict the notes of one instrument from a two instrument audio file.

## Specifications

Currently using a simple Categorical Neural Network built using keras.
The melodies used for building training data were built by generating sequences of randomized notes from a random major key based off of a note between C4 and C5 (which in total covers 24 notes or two octaves).
Each wav file should be 12 seconds long, wherein each second is comprised of the MIDI recording of two random notes, one in the soundfont of a grand piano, the other in the soundfont of a flute.
Also currently using a basic static web template. Will be used in a web application. Yields a 74% accuracy, and using a boosted simple random forest gave similar accuracy.

## How to Use

The function main_function in access.py, takes in an audio file (a wav file where there are by default twelve seconds of audio) and outputs the notes of the piano.
You can also access the trained model directly (categorical_model).
