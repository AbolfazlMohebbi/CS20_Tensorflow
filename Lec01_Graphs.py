import tensorflow as tf


#################### Distributed Computation #####################
# To put part of a graph on a specific CPU or GPU

# Creates a graph.
with tf.device('/gpu:0'):  # or on GPU: with tf.device('/gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
    c = tf.multiply(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

#################### Graph Computation #####################

# 1-Multiple graphs require multiple sessions, each will try to use all available resources by default
# 2-Can't pass data between them without passing them through python/numpy, which doesn't work in distributed
# 3-It’s better to have disconnected subgraphs within one graph

g = tf.Graph()
with g.as_default(): # to add operators to a graph, set it as default
    x = tf.add(3, 5)
sess2 = tf.Session(graph=g)
with tf.Session() as sess2:
    print("heyyyyyyyyy")
    sess2.run(x)
