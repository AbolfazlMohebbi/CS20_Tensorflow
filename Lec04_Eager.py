# EAGER is a NumPy-like library for numerical computation with
# support for GPU acceleration and automatic
# differentiation, and a flexible platform for machine
# learning research and experimentation

import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
import tools

DATA_FILE = 'data/birth_life_2010.txt'

tfe.enable_eager_execution()

data, n_samples = tools.readfile(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))

# Create variables.
w = tfe.Variable(0.0)
b = tfe.Variable(0.0)

# Define the linear predictor.
def prediction(x):
    return x * w + b

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
    return (y - y_predicted) ** 2

def huber_loss(y, y_predicted, m=1.0):
    diff = y - y_predicted
    return diff ** 2 if tf.abs(diff) <= m else m * (2 * tf.abs(diff) - m)

def train(loss_fn):
    print('Training; loss function: ' + loss_fn.__name__)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # Define the function through which to differentiate.
    def loss_for_example(x, y):
        return loss_fn(y, prediction(x))

    grad_fn = tfe.implicit_value_and_gradients(loss_for_example)

    n_epoch = 100

    for epoch in range(n_epoch):
        total_loss = 0.0
        for x_i, y_i in tfe.Iterator(dataset):
            loss, gradients = grad_fn(x_i, y_i)
            # Take an optimization step and update variables.
            optimizer.apply_gradients(gradients)
            total_loss += loss
        if epoch % 10 == 0:
            print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))

train(huber_loss)
plt.plot(data[:,0], data[:,1], 'bo')
# The `.numpy()` method of a tensor retrieves the NumPy array backing it.
# In future versions of eager, you won't need to call `.numpy()` and will
# instead be able to, in most cases, pass Tensors wherever NumPy arrays are
# expected.
plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r', label="huber regression")
plt.legend()
plt.show()
