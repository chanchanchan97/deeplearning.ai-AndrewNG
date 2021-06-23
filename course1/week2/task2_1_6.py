# GRADED FUNCTION: L1
import numpy as np

def L1(yhat, y):
    x = yhat - y
    loss = np.dot(x, x)
    
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
