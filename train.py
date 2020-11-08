#source file for training

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

#read names from file
names = np.genfromtxt("train/names_labels.txt", dtype=str, delimiter = ',', skip_header = 1, usecols = (0))
#read labels from file
labels = np.genfromtxt('train/names_labels.txt', delimiter = ',', skip_header = 1, usecols = (1))

#read onehots from file
data = np.load('train/names_onehots.npy', allow_pickle = True).item()
onehots = data['onehots']

#read validation labels from file
validLabels = np.genfromtxt('validation/names_labels.txt', delimiter = ',', skip_header = 1, usecols = (1))

#read validation onehots from file
validData = np.load('validation/names_onehots.npy', allow_pickle = True).item()
validOnehots = validData['onehots']

#get size of 2D matrix of onehots
size = onehots[0].shape
#set input shape according to the size
inputShape = (1, 73, 398)

#check size for reshape purpose
#print(onehots.shape)
#print(validOnehots.shape)

#reshape onehots for input purpose
onehots = np.reshape(onehots, [8169, 1, 73, 398])
validOnehots = np.reshape(validOnehots , [272, 1, 73, 398])

#create model
model = models.Sequential()

#add layers to the model
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape = inputShape, padding = 'same'))
model.add(layers.MaxPooling2D((2, 2), padding = 'same'))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2), padding = 'same'))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'))

#flatten 3D output to 1D for dense layers
model.add(layers.Flatten())
#add dense layers to perform classification
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

#compile the model
model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#iteration of training
history = model.fit(onehots, labels, epochs=10, 
                    validation_data=(validOnehots, validLabels))

#save the weights
model.save('weights/savedWeights')
