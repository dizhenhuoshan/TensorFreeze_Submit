""" import your model here """
import your_model as tf
""" your model should support the following code """

# create model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

W_grad = tf.gradients(cross_entropy, [W])[0]
train_step = tf.assign(W, W - 0.5 * W_grad)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# get the mnist dataset (use tensorflow here)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# eval
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

ans = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

print("Accuracy: %.3f" % ans)
assert ans >= 0.87
