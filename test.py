#source file for recovering the network model

import numpy as np
import tensorflow as tf

#read onehots from file
data = np.load('test/names_onehots.npy', allow_pickle = True).item()
onehots = data['onehots']
