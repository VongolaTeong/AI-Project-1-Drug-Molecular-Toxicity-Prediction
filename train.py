#source file for training

import numpy as np

#read names from file
names = np.genfromtxt("train/names_labels.txt", dtype=str, delimiter = ',', skip_header = 1, usecols = (0))
#read labels from file
labels = np.genfromtxt('train/names_labels.txt', delimiter = ',', skip_header = 1, usecols = (1))
