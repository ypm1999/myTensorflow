#!/usr/bin/env python3
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

class LogisticRegression:
    def __init__(self, n, m, step_len = 0.01):
        #n = 784, m = 10
        self.n = n
        self.m = m
        self.w = np.ones((n, m), dtype=np.float)
        self.b = np.zeros((1, m), dtype=np.float)
        self.step = step_len

    def softmax(self, x):
        tmp = np.exp(-x)
        return tmp / np.sum(tmp)

    def run(self, data, std):
        dw = np.zeros((self.n, self.m), dtype=np.float)
        db = np.zeros((1, self.m), dtype=np.float)
        n = len(data)
        for i in range(n):
            x_now = np.mat(data[i])
            y_now = np.mat(std[i])
            a = self.softmax(np.dot(x_now, self.w) + self.b)
            dz = a - y_now
            db += dz
            dw += np.dot(x_now.T, dz)
        dw = dw * self.step
        db = db * self.step
        self.w += dw
        self.b += db
        #self.step *= 0.999


    def predication(self, data):
        #print(self.softmax(np.dot(data, self.w) + self.b))
        result = []
        for x in data:
            result.append(np.argmax(self.softmax(np.dot(x, self.w) + self.b)))
        return result


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    model = LogisticRegression(28 * 28, 10, 0.01)
    for i in range(1000):
        x_batch, y_batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            tmp = np.equal(np.argmax(y_batch, axis=1), model.predication(x_batch))
            accuracy = np.mean(tmp.astype(np.float))
            print("step %d, training accuracy %g" % (i, accuracy))
        model.run(x_batch, y_batch)
    x_batch, y_batch = mnist.test.next_batch(5000)
    tmp = np.equal(np.argmax(y_batch, axis=1), model.predication(x_batch))
    accuracy = np.mean(tmp.astype(np.float))
    print("Final training accuracy %g" % (accuracy))


if __name__ == "__main__":
    main()

