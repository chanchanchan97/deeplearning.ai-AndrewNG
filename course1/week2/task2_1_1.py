# GRADED FUNCTION: basic_sigmoid
import math
import numpy as np

def basic_sigmoid(x):
    s = 1/(1 + math.exp(-x))
    
    return s

print(basic_sigmoid(3))

def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    
    return s

x = np.array([1, 2, 3])
print(sigmoid(x))
