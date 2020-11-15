#source file for recovering the network model

import numpy as np
import tensorflow as tf
import os
import statistics

seed_value = 100
tf.set_random_seed(seed_value)

def load_test_data_name(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    onehots = data['onehots']
    name = data['names']
    return onehots, name

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1_layer = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu)
        self.pool1_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.conv2_layer = tf.keras.layers.Conv2D(32, 3, (1, 2), 'same', activation=tf.nn.relu)
        self.pool2_layer = tf.keras.layers.MaxPool2D(2, 2)
        # flat
        self.FCN = tf.keras.layers.Dense(2)
        # softmax

    def call(self, inputs):
        x = self.conv1_layer(inputs)
        x = self.pool1_layer(x)
        x = self.conv2_layer(x)
        x = self.pool2_layer(x)
        flat = tf.reshape(x, [-1, 18*50*32])
        output = self.FCN(flat)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm
    
# data
test_path = os.path.join(os.path.dirname(__file__), "test/")
test_data, test_name = load_test_data_name(test_path)
name = test_name

# model
tf.reset_default_graph()  # 
model = MyModel()
input_place_holder = tf.placeholder(tf.float32, [None] + list(test_data.shape[1:]), name='input')
input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + list(test_data.shape[1:]) + [1])
output, output_with_sm = model(input_place_holder_reshaped)

# Predict on the test set
data_size = test_data.shape[0]
BatchSize = 128
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'weights/weights')
    prediction = []
    for i in range(0, data_size, BatchSize):
        print(i)
        test_output = sess.run(output, {input_place_holder: test_data[i:i + BatchSize]})
        test_output_with_sm = sess.run(output_with_sm, {input_place_holder: test_data[i:i + BatchSize]})
        pred = test_output_with_sm[:, 1]
        prediction.extend(list(pred))
sess.close()
f = open('output_518030990014.txt', 'w')
f.write('Chemical,Label\n')

mean = statistics.mean(prediction)
for x, i in enumerate(prediction):
    if prediction[x] > mean:
        prediction[x] = 1
    else:
        prediction[x] = 0

i = 0
for x in prediction:
    f.write(name[i] + ',' + str(x) + '\n')
    i += 1
f.close()
