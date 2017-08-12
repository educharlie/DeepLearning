
# coding=utf-8
import numpy as np 

def sigmoid(x):
   
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    
    s = sigmoid(x)
    return s*(1-s)

def image2vector(image):
    
    return image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))

def normalizeRows(x):

    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    return x/x_norm

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis = 1, keepdims = True)
    return x_exp/x_sum

x = np.array([1, 2, 3])
print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))

x = np.array([
    [0, 3, 4],
    [2, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))