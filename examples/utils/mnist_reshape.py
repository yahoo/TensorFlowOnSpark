# Copyright 2019 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

import sys
import numpy as np
vec = [int(x) for x in next(sys.stdin).split(',')]
img = np.reshape(vec[1:], (28, 28, 1))
print(np.array2string(img).replace('\n ', ','))
