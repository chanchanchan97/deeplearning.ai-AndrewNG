# GRADED FUNCTION: normalizeRows
import numpy as np

def normalizeRows(x):
    A = np.linalg.norm(x, axis = 1, keepdims = True)
    x_norm = x / A

    return x_norm

x = np.array([[0, 3, 4], [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))
