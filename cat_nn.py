from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense #, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
print('imported everything')

# dropouts
# 2 se more not imp
# building our model
def build_model():
	model = Sequential()
	model.add(Dense(128, input_shape=[128]))
	model.add(Dense(128, activation='linear'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(25, activation='softmax'))

	optimizer = Adam(lr)

	model.compile(loss='categorical_crossentropy',
		optimizer=optimizer,
		metrics=['accuracy'])

	return model

def shuffle_in_unison(x,y):
	i = np.arange(0,len(x),1)
	np.random.shuffle(i)
	return x[i], y[i]

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
		results.append(n)
	return results, counts

#set variables
outp_PATH = 'D:\\y_data.npy'
inp_PATH =  'D:\\x_data.npy'
total_inputs = 8478*12
train_split = 0.8
train_no = int(total_inputs*train_split)
lr = 0.005
EPOCHS = 10
BATCH = 256
print('Set vars')

x_raw = np.load(inp_PATH)
y_raw = (np.load(outp_PATH))/10
# one hot encoding
y_oh = to_categorical(y_raw-59, num_classes=25)

x, y = shuffle_in_unison(x_raw, y_oh)
#split into train and test
x_train, x_test = np.split(x, [train_no])
y_train, y_test = np.split(y, [train_no])
print('split data')

# our neural network
"""model = build_model()
model.summary()"""

model = load_model('categorical_model')
model.summary()

# training
history = model.fit(
  x_train, y_train,
  epochs=EPOCHS, validation_data = (x_test, y_test), verbose=2,
  batch_size=BATCH)

# visualizing model and performance
train_acc = history.history['acc']
val_acc = history.history['val_acc']

np.save('cat_train_start.npy', train_acc)
np.save('cat_val_start.npy', val_acc)

epoch_count = range(1, len(train_acc) + 1)
plt.ylim(0, 1)
plt.plot(epoch_count, train_acc, 'b-')
plt.plot(epoch_count, val_acc, 'r-')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.show()

#save model
#model.save('categorical_model')

# personal stats check
"""predictions = model.predict([x_test])
print(predictions.shape)
results, counts = rev_one_hot(predictions)
expr, expc = rev_one_hot(y_test)
print(results[:10])
print(expr[:10])
print(counts)
print(expc)
print(np.asarray(counts)-np.asarray(expc))
"""