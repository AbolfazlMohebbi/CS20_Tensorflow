import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # No more warnings
import tensorflow as tf

a = tf.zeros([2, 1], tf.int32)
a = tf.ones([2, 1], tf.int32)
a = tf.fill([2, 1], 19)
a = tf.constant([[12],[10]], name="Constant_A")
b = tf.zeros_like(a)
b = tf.constant([[4],[7]], name='Constant_B')
x = tf.add(a, b, name='add')
writer = tf.summary.FileWriter('./TensorboardLogs', tf.get_default_graph())
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))
writer.close() # close the writer when you’re done using it
# At Terminal: tensorboard --logdir="./TensorbordLogs" --port 6006


varA = tf.get_variable("newMatrix", initializer=tf.ones([2, 3], tf.int32))
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    vA = sess2.run(varA)
    print(vA)
    print(varA.eval())
