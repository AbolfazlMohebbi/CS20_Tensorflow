import numpy as np
import tensorflow as tf

def readfile(filename):
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples

def huber_loss_tf(y_ph, yHat_ph, delta=14.0):
    # inputs are placeholders so loss is also placeholder
    diff = tf.abs(y_ph-yHat_ph, name = 'diff')
    def f1(): return (0.5 * tf.square(diff))
    def f2(): return (delta * diff - 0.5 * tf.square(delta))
    loss = tf.cond(tf.less(diff, delta), f1, f2)
    return loss


def huber_loss(y, y_hat, delta):
    diff = abs(y - y_hat)
    if diff <= delta:
        loss = 0.5 * (diff**2)
    else:
        loss = (delta * diff) - (0.5 * delta**2)

    return loss


