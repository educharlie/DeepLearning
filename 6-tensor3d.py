import tensorflow as tf
import numpy as np

tensor_3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(tensor_3d)
print(tensor_3d.shape)

#tensor_3d[plane,row,col]
print(tensor_3d[1,0,1]) #6