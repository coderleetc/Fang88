"""
这是一段简单的神经网络代码，结构是10层网络。想分享这段代码是为了表现我的代码结构，思路和逻辑是相对清晰的，希望您认可。
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_train = pd.read_csv('D:\\Work\\Work for Fang88\\Data analysis Intern\\Project1\\dataset1\\train.csv')
data_test = pd.read_csv('D:\\Work\\Work for Fang88\\Data analysis Intern\\Project1\\dataset1\\test.csv')
data_tobe = pd.read_csv('D:\\Work\\Work for Fang88\\Data analysis Intern\\Project1\\dataset1\\tobe.csv')

x_data_test = data_test.iloc[:, 0:19].as_matrix()
y_data_test = data_test.iloc[:, 19:20].as_matrix()
x_data_train = data_train.iloc[:, 0:19].as_matrix()
y_data_train = data_train.iloc[:, 19:20].as_matrix()
x_data_tobe = data_tobe.iloc[:, 0:19].as_matrix()

xs_test = tf.convert_to_tensor(np.float32(x_data_test))
ys_test = tf.convert_to_tensor(np.float32(y_data_test))
xs_train = tf.convert_to_tensor(np.float32(x_data_train))
ys_train = tf.convert_to_tensor(np.float32(y_data_train))
xs_tobe = tf.convert_to_tensor(np.float32(x_data_tobe))



def train():
    print('Training...')
    
    tf_x = tf.placeholder(tf.float32, x_data_train.shape)
    tf_y = tf.placeholder(tf.float32, y_data_train.shape)
    
    l1 = tf.layers.dense(tf_x, 30, tf.nn.relu)
    l2 = tf.layers.dense(l1, 30, tf.nn.relu)
    l3 = tf.layers.dense(l2, 30, tf.nn.relu)
    l4 = tf.layers.dense(l3, 30, tf.nn.relu)
    l5 = tf.layers.dense(l4, 30, tf.nn.relu)
    l6 = tf.layers.dense(l5, 30, tf.nn.relu)
    l7 = tf.layers.dense(l6, 30, tf.nn.relu)
    l8 = tf.layers.dense(l7, 30, tf.nn.relu)
    l9 = tf.layers.dense(l8, 30, tf.nn.relu)
    l10 = tf.layers.dense(l9, 30, tf.nn.relu)
    o = tf.layers.dense(l10, 1)
    
    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(1 - (o / tf_y)), reduction_indices=[1]))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for step in range(5000000):
        sess.run(train_op, {tf_x: x_data_train, tf_y: y_data_train})
        print(sess.run(loss, feed_dict={tf_x: x_data_train, tf_y: y_data_train}))
        print('training step' + str(step))
    saver.save(sess, 'D:\\Work\\Work for Fang88\\Data analysis Intern\\Project1\\my_model', write_meta_graph=False)  # meta_graph is not recommended


def test():
    print('This is test')
    
    tf_x = tf.placeholder(tf.float32, x_data_test.shape)
    tf_y = tf.placeholder(tf.float32, y_data_test.shape)
    
    l1_ = tf.layers.dense(tf_x, 30, tf.nn.relu)
    l2_ = tf.layers.dense(l1_, 30, tf.nn.relu)
    l3_ = tf.layers.dense(l2_, 30, tf.nn.relu)
    l4_ = tf.layers.dense(l3_, 30, tf.nn.relu)
    l5_ = tf.layers.dense(l4_, 30, tf.nn.relu)
    l6_ = tf.layers.dense(l5_, 30, tf.nn.relu)
    l7_ = tf.layers.dense(l6_, 30, tf.nn.relu)
    l8_ = tf.layers.dense(l7_, 30, tf.nn.relu)
    l9_ = tf.layers.dense(l8_, 30, tf.nn.relu)
    l10_ = tf.layers.dense(l9_, 30, tf.nn.relu)
    o_ = tf.layers.dense(l10_, 1)
    
    loss_ = tf.reduce_mean(tf.reduce_sum(tf.abs(1 - (o_ / tf_y)), reduction_indices=[1]))

    sess = tf.Session()

    saver = tf.train.Saver()
    savePath = 'D:\\Work\\Work for Fang88\\Data analysis Intern\\Project1\\my_model'
    saver.restore(sess, savePath)
    print('Trained model is saved to ' + savePath)

    print("Test accuracy is " + str(sess.run(loss_, feed_dict={tf_x: x_data_test, tf_y: y_data_test})))

def predict():
    print('Predicting...')

    tf_x = tf.placeholder(tf.float32, x_data_tobe.shape)
    l1__ = tf.layers.dense(tf_x, 30, tf.nn.relu)
    l2__ = tf.layers.dense(l1__, 30, tf.nn.relu)
    l3__ = tf.layers.dense(l2__, 30, tf.nn.relu)
    l4__ = tf.layers.dense(l3__, 30, tf.nn.relu)
    l5__ = tf.layers.dense(l4__, 30, tf.nn.relu)
    l6__ = tf.layers.dense(l5__, 30, tf.nn.relu)
    l7__ = tf.layers.dense(l6__, 30, tf.nn.relu)
    l8__ = tf.layers.dense(l7__, 30, tf.nn.relu)
    l9__ = tf.layers.dense(l8__, 30, tf.nn.relu)
    l10__ = tf.layers.dense(l9__, 30, tf.nn.relu)
    o__ = tf.layers.dense(l10__, 1) 

    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, 'D:\\Work\\Work for Fang88\\Data analysis Intern\\Project1\\my_model')

    prediction = sess.run(o__, feed_dict={tf_x: x_data_tobe})
    savePath = 'D:\\Work\\Work for Fang88\\Data analysis Intern\\Project1\\output.csv'
    np.savetxt(savePath, prediction, delimiter = ",")
    print('Prediction has been saved to ' + savePath)
    
train()

tf.reset_default_graph()
test()

tf.reset_default_graph()  
predict()
