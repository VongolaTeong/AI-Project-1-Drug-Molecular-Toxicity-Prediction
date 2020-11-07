#source file for training

import numpy as np

#read names from file
names = np.genfromtxt("train/names_labels.txt", dtype=str, delimiter = ',', skip_header = 1, usecols = (0))
#read labels from file
labels = np.genfromtxt('train/names_labels.txt', delimiter = ',', skip_header = 1, usecols = (1))

#read onehots from file
data = np.load('train/names_onehots.npy', allow_pickle = True).item()
onehots = data['onehots']

