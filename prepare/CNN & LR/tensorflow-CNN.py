#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as data


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape = shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options = gpu_options)
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)

    mnist = data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.InteractiveSession(config=config)
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    conv1_W = weight_variable([5, 5, 1, 32])
    conv1_b = bias_variable([32])
    image_x = tf.reshape(x, [-1, 28, 28, 1])
    conv1_h = tf.nn.relu(conv2d(image_x, conv1_W) + conv1_b)
    pool1_h = max_pool_2x2(conv1_h)

    conv2_W = weight_variable([5, 5, 32, 64])
    conv2_b = bias_variable([64])
    conv2_h = tf.nn.relu(conv2d(pool1_h, conv2_W) + conv2_b)
    pool2_h = max_pool_2x2(conv2_h)

    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(pool2_h, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([1024, 10])
    b_fc2 = weight_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())

    for i in range(100):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    batch = mnist.test.next_batch(8000)
    print(sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))


def main2():
    mnist = data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    batch = mnist.test.next_batch(1000)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]}))

if __name__ == "__main__":
    main()
