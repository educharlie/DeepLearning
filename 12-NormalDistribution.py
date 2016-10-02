import tensorflow as tf
import matplotlib.pyplot as plt

norm = tf.random_normal([100],mean = 0, stddev = 2)

with tf.Session() as session:
	normal_eval = norm.eval()

print(normal_eval)
plt.hist(normal_eval, normed = True)
plt.show()