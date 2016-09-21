import numpy as np
import tensorflow as tf

tensor_1d = np.array([1.3, 1, 4.1, 23.99])

#convert python array to tensor
tf_tensor = tf.convert_to_tensor(tensor_1d, dtype = tf.float64)

with tf.Session() as sess:
	print(sess.run(tf_tensor))
	print(sess.run(tf_tensor[0]))
	print(sess.run(tf_tensor[1]))


tensor_2d = np.array([(1,2,3,4),(3,1,4,2),(6,2,6,3),(9,0,5,2)])
print(tensor_2d[2][3])

