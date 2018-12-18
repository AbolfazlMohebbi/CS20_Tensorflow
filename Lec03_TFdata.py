import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tools
import tensorflow as tf
import matplotlib.pyplot as plt

DATA_FILE = 'data/birth_life_2010.txt'

# Step 1: read in the data
data, n_samples = tools.readfile(DATA_FILE)

dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
iterator = dataset.make_initializable_iterator()


birth_ph, life_ph = iterator.get_next()  # All the rows, one-by-one, all as tensors

theta0 = tf.get_variable('theta0', initializer=tf.constant(0.0))
theta1 = tf.get_variable('theta1', initializer=tf.constant(0.0))

life_predicted = theta1 * birth_ph + theta0

# loss = tf.square(life_ph - life_predicted, name='loss')
loss = tools.huber_loss_tf(life_ph, life_predicted)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


n_epochs = 100
birth = data[:, 0]
life = data[:, 1]

with tf.Session() as Sess:
    Sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./TensorboardLogs/linear_reg', Sess.graph)

    for i in range(n_epochs):
        total_loss = 0
        Sess.run(iterator.initializer) # initialize the iterator
        try:
            while True:
                _, loss_iter = Sess.run([optimizer, loss])
                total_loss += loss_iter
        except tf.errors.OutOfRangeError:
            pass

        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    writer.close()
    theta0_out, theta1_out = Sess.run([theta0, theta1])
    #life_pre_out = Sess.run(life_predicted)

plt.plot(birth, life, 'bo', label='Real data')
plt.plot(birth, theta0_out + theta1_out * birth , 'r', label='Predicted data')
plt.legend()
plt.show()
