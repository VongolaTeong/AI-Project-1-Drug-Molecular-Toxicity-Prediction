#source file for training

import numpy as np
import tensorflow as tf
import pandas as pd
import os

#set a constant seed value to get consistent results
seed_value = 100
tf.compat.v1.set_random_seed(seed_value)

#function to load datasets
def load_data(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    data = data['onehots']
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    return data, label

#model class
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        #set layers
        self.conv1_layer = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu)
        self.pool1_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.conv2_layer = tf.keras.layers.Conv2D(32, 3, (1, 2), 'same', activation=tf.nn.relu)
        self.pool2_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.flatten_layer = tf.keras.layers.Flatten()
        #flat
        self.FCN = tf.keras.layers.Dense(2)
        #softmax

    def call(self, inputs):
        x = self.conv1_layer(inputs)
        x = self.pool1_layer(x)
        x = self.conv2_layer(x)
        x = self.pool2_layer(x)
        flat = self.flatten_layer(x)
        output = self.FCN(flat)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm

def run():
    #parameters
    LR = 0.01
    BatchSize = 128
    EPOCH = 15

    train_data_path = os.path.join(os.path.dirname(__file__), "train/")
    validation_data_path = os.path.join(os.path.dirname(__file__), "validation/")

    #data
    train_x, train_y = load_data(train_data_path)
    valid_x, valid_y = load_data(validation_data_path)

    #create model instance
    model = MyModel()

    #input and output
    onehots_shape = list(train_x.shape[1:])
    input_place_holder = tf.compat.v1.placeholder(tf.float32, [None] + onehots_shape, name='input')
    input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + onehots_shape + [1])
    label_place_holder = tf.compat.v1.placeholder(tf.int32, [None], name='label')
    label_place_holder_2d = tf.one_hot(label_place_holder, 2)
    output, output_with_sm = model(input_place_holder_reshaped)

    #show model's structure
    model.summary() 

    #loss
    bce = tf.keras.losses.BinaryCrossentropy()
    loss = bce(label_place_holder_2d, output_with_sm)

    #optimizer
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    #auc
    prediction_place_holder = tf.compat.v1.placeholder(tf.float64, [None], name='pred')
    auc, update_op = tf.compat.v1.metrics.auc(labels=label_place_holder, predictions=prediction_place_holder)

    #run
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        saver = tf.compat.v1.train.Saver()

        train_size = train_x.shape[0]
        best_val_auc = 0
        
        for epoch in range(EPOCH):
            for i in range(0, train_size, BatchSize):
                b_x, b_y = train_x[i:i + BatchSize], train_y[i:i + BatchSize]
                _, loss_ = sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y})

                print("Epoch {}: [{}/{}], training set loss: {:.4}".format(epoch, i, train_size, loss_))

            if epoch % 1 == 0:
                val_prediction = sess.run(output_with_sm, {'input:0': valid_x})
                val_prediction = val_prediction[:, 1]
                auc_value = sess.run(update_op, feed_dict={prediction_place_holder: val_prediction, label_place_holder: valid_y})
                print("auc_value", auc_value)
                if auc_value > best_val_auc:
                    saver.save(sess, 'weights/weights')
                    
if __name__ == "__main__":
    run()
