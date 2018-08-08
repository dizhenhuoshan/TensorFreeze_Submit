""" import your model here """
import your_model as tf
""" your model should support the following code """

import numpy as np

sess = tf.Session()

# linear model
W = tf.Variable([.5], dtype=tf.float32)
b = tf.Variable([1.5], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

# define error
y = tf.placeholder(tf.float32)
error = tf.reduce_sum(linear_model - y)

# run init
init = tf.global_variables_initializer()
sess.run(init)

# calc error
feed = {x: [1,2,3,4], y: [0, -1, -2, -3]}

# assign
fixW = tf.assign(W, [-1.0])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
ans = sess.run(error, feed)

assert np.equal(ans, 0)
