
import numpy as np


A = np.array([ (1,0,0) ,(2, 2, 2),(4,8,16)])  # Corrected the last row
Y = np.array([ 1,6,20])

A_inv = np.linalg.inv(A)

# menyelesaikan persamaan linear
X1 = np.dot(A_inv,Y)
print(X1)




