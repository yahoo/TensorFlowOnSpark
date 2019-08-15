import sys
import numpy as np
vec = [int(x) for x in next(sys.stdin).split(',')]
img = np.reshape(vec[1:], (28, 28, 1))
print(np.array2string(img).replace('\n ', ','))
