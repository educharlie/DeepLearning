import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#complex numbers
x = 5. + 4j
print(x)
y = complex(5,4)
print(y)
print(y.real , y.imag)

#fractasl and madelbrot set
#Complex plane between -1.3 to 1-3 in the real axis and between -2j ans 1j on the imaginary axis
Y, X =np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X + 1j * Y
c = tf.constant(Z.astype(np.complex64))

zs = tf.Variable(c)
ns = tf.Variable(tf.zeros_like(c, tf.float32))
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
zs_ = zs * zs + c
not_diverged = tf.complex_abs(zs_) < 4

step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)))
for i in range(200): step.run()

plt.imshow(ns.eval())
plt.show()
