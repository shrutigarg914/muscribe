from tensorflow.keras.models import load_model
from functions import load, process_track, combine
import numpy as np
from librosa.output import write_wav

# take audio and return 1 second snippets
def process_audio(wav_file, length_in_seconds=12, ):
	array = process_track(load(wav_file))
	n = array.shape[1]/length_in_seconds
	temp = []
	tru_arr = np.transpose(array) # because there are 128 arrays of frequencies for each timestamp we want arrays of 128 frequencies for each timestamp
	for i in range(0,length_in_seconds):
		temp.append(tru_arr[int(10+(n*i))])
	return np.asarray(temp)

# load model only during function
def predict(array):
	model = load_model('categorical_model')
	return model.predict(array)

# functional helpers
def rev_one_hot(array):
	i = 0
	results = []
	counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for j in range(0, array.shape[0]):
		i +=1
		a = array[j]
		n = np.argmax(a)
		if n>24:
			print(i)
			n = 25
		counts[n] += 1
		results.append(n+59)
	return results, counts

def convert_to_notes(array, start=60, end=84):
	names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
	octave = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 24 is C1, 36, 48 octave 1 - 
	result = []
	for i in array:
		r = int(i) % 12
		name = names[r]
		no = octave[int(i/12)-1]
		note = name+str(no)
		result.append(note)
	return result

def generate(piano, flute, out_path):
	write_wav(out_path, combine(load(piano), load(flute)), 40000)

# take file and return note names of piano track
def main_function(file_path, seconds=12):
	results, counts = rev_one_hot(predict(process_audio(file_path, seconds)))
	return convert_to_notes(results)
"""
generate("D:\\MIDI_demo\\1.wav", "D:\\MIDI_demo\\2.wav", 'D:\\MIDI_demo\\test.wav')

print(main_function('D:\\MIDI_demo\\test.wav'))
"""