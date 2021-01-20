#source file for training

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.utils import plot_model

#set a constant seed value to get consistent results
seed_value = 100
tf.compat.v1.set_random_seed(seed_value)

def createModel(size):

    #create model
    model = models.Sequential()

    #add layers to the model
    model.add(layers.Dense(1024, input_shape = size))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))          

    #compile the model
    model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['AUC'])
    
    return model

def run():
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

    model = createModel(size)
    
    model.summary()
    plot_model(model, show_shapes=True, to_file='FCN_train.png')
    
    checkpoint_path = "FCN_weights/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                 save_weights_only = True,
                                                 verbose = 1)

    #train the model with the new callback
    model.fit(onehots, labels, epochs = 2, validation_data=(validOnehots, validLabels),
          callbacks=[cp_callback])  #pass callback to training

    
    
if __name__ == "__main__":
    run()
