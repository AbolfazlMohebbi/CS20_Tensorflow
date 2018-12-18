import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # No more warnings

# ================== Variables are Python Arrays ====================
x = [2.3, 2.9, 2.4, 1.9]
y = [1.1, 1.3, 1.9, 1.5]

sum = tf.add(x, y)

with tf.Session() as sess_1:
    res = sess_1.run(sum)
    print('Sum of Arrays - Variables are Python Arrays')
    print(res)

# ================== Variables are Tensorflow stuff =====================
a = tf.constant([2.3, 2.9, 2.4, 1.9], dtype=tf.float32, name = "cte_a")
b = tf.Variable([1.1, 1.3, 1.9, 1.5], dtype=tf.float32, name = "var_b")

sum_tf = tf.add(a, b, name = "sumTF")

with tf.Session() as sess_tf:
    sess_tf.run(tf.global_variables_initializer()) # Here we must initialize tensors
    res_tf = sess_tf.run(sum_tf)
    print('Sum of Arrays - Variables are Tensorflow stuff')
    print(res_tf)

# ================== Variables using placeholders ===================
p = tf.placeholder(dtype=tf.float32, shape=(1,4), name = "var_p")
q = tf.placeholder(dtype=tf.float32, shape=(1,4), name = "var_q")

sum_ph = tf.add(p, q)

with tf.Session() as sess_ph:
    res_ph = sess_ph.run(sum_ph, feed_dict={p:[[2.3, 2.9, 2.4, 1.9]], q:[[1.1, 1.3, 1.9, 1.5]]})  # Notice the double bracket
    print('Sum of Arrays - Variables are placeholders')
    print(res_ph)

# Shapes
# p:[2.3, 2.9, 2.4, 1.9]  ==> shape: (4,)
# p:[[2.3, 2.9, 2.4, 1.9]] ==> shape: (1,4)
# p:[[2.3], [2.9], [2.4], [1.9]] ==> shape: (4,1)


