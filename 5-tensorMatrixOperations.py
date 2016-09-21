import tensorflow as tf
import numpy as np

matrix1 = np.array([(2,2,2),(2,2,2),(2,2,2)], dtype = 'int32')
matrix2 = np.array([(1,1,1),(1,1,1),(1,1,1)], dtype = 'int32')

#transforming into a tensor

matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)

matrix_product = tf.matmul(matrix1, matrix2)
matrix_sum = tf.add(matrix1, matrix2)

matrix3 = tf.constant(np.array([(1,2,3),(4,2,5),(7,3,1)], dtype = 'float32'))
matrix_det = tf.matrix_determinant(matrix3)

with tf.Session() as sess:
	print(sess.run(matrix_product))
	print(sess.run(matrix_sum))
	print(sess.run(matrix_det))

