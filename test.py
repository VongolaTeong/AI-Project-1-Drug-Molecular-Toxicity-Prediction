#source file for recovering the network model

import numpy as np
import tensorflow as tf
import os

#set a constant seed value to get consistent results
seed_value = 100
tf.compat.v1.set_random_seed(seed_value)

#function to load datasets
def load_test_data_name(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    onehots = data['onehots']
    name = data['names']
    return onehots, name

#model class
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        #set layers
        self.conv1_layer = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu)
        self.pool1_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.conv2_layer = tf.keras.layers.Conv2D(64, 3, (1, 2), 'same', activation=tf.nn.relu)
        self.pool2_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.FCN = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = self.conv1_layer(inputs)
        x = self.pool1_layer(x)
        x = self.conv2_layer(x)
        x = self.pool2_layer(x)
        flat = self.flatten_layer(x)
        output = self.FCN(flat)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm
    
#data
test_path = os.path.join(os.path.dirname(__file__), "test/")
test_data, test_name = load_test_data_name(test_path)
name = test_name

#model
tf.compat.v1.reset_default_graph() 
model = MyModel()
input_place_holder = tf.compat.v1.placeholder(tf.float32, [None] + list(test_data.shape[1:]), name='input')
input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + list(test_data.shape[1:]) + [1])
output, output_with_sm = model(input_place_holder_reshaped)

#predict on the test set
data_size = test_data.shape[0]
BatchSize = 128
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, 'weights/weights')
    prediction = []
    for i in range(0, data_size, BatchSize):
        #print(i)
        test_output = sess.run(output, {input_place_holder: test_data[i:i + BatchSize]})
        test_output_with_sm = sess.run(output_with_sm, {input_place_holder: test_data[i:i + BatchSize]})
        pred = test_output_with_sm[:, 1]
        prediction.extend(list(pred))
        
sess.close()

f = open('output_518030990014.txt', 'w')
f.write('Chemical,Label\n')

for i, v in enumerate(prediction):
        f.write(name[i] + ',%f\n' % v)
    
f.close()
