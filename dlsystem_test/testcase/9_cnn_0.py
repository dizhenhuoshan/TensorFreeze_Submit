""" import your model here """
import your_model as act_tf
import tensorflow as ans_tf
""" check the correctness of your model by comparing with tensorflow """

import numpy as np

def test_forward_conv(tf):
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[3, 4, 4, 2])
        w = tf.placeholder(tf.float32, shape=[3, 3, 2, 5])

        conv1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        res = sess.run(conv1, feed_dict={
            x: np.arange(3*4*4*2).reshape(3, 4, 4, 2),
            w: np.arange(3*3*2*5).reshape(3, 3, 2, 5),
            })
        return res

def test_backward_conv(tf):
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[3, 4, 4, 2])
        w = tf.Variable(np.arange(3*3*2*5, dtype=np.float32).reshape(3, 3, 2, 5))

        conv1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

        loss = tf.reduce_sum(conv1)
        train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)

        sess.run(tf.global_variables_initializer())
        sess.run(train_op, feed_dict={
            x: np.arange(3*4*4*2).reshape(3, 4, 4, 2),
        })

        res = w.eval()
        return res

def test_forward_maxpool(tf):
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[3, 4, 4, 2])
        pool1 = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
            padding='SAME')

        res =sess.run(pool1, feed_dict={
            x: np.arange(3*4*4*2).reshape(3, 4, 4, 2),
        })
        return res

def test_backward_maxpool(tf):
    with tf.Session() as sess:
        x = tf.Variable(np.arange(3*4*4*2, dtype=np.float32).reshape(3, 4, 4, 2))
        pool1 = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
            padding='SAME')
    

        loss = tf.reduce_sum(pool1)
        train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)

        sess.run(tf.global_variables_initializer())
        sess.run(train_op)

        res = x.eval()
        return res

if __name__ == "__main__":
    testcases = [
            test_forward_conv,
            test_backward_conv,
            test_forward_maxpool,
            test_backward_maxpool
    ]

    for item in testcases:
        res = item(act_tf)
        ans = item(ans_tf)

        np.testing.assert_allclose(res, ans, atol=1e-2)

