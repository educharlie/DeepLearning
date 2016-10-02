import tensorflow as tf
import matplotlib.pyplot as plt

uniform = tf.random_uniform([100],minval = 0, maxval = 1, dtype = tf.float32)

with tf.Session() as session:
	uniform_eval = uniform.eval()

print(uniform_eval)
plt.hist(uniform_eval, normed = True)
plt.show()