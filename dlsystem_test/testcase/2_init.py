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

init = tf.global_variables_initializer()
sess.run(init)

ans = sess.run(linear_model, {x: [1,2,3,4]})
assert np.array_equal(ans, [2, 2.5, 3, 3.5])
