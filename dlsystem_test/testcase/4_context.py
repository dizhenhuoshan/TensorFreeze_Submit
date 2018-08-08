""" import your model here """
import your_model as tf
""" your model should support the following code """

import numpy as np

# adder 
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b;

"""
your session should support 'with' statement
see http://blog.csdn.net/largetalk/article/details/6910277
"""
with tf.Session() as sess:
    ans = sess.run(adder_node, {a: 3, b: 4.5})
    assert np.equal(ans, 7.5)

    ans = sess.run(adder_node, {a: [1, 3], b: [2, 3]})
    assert np.array_equal(ans, [3, 6])
