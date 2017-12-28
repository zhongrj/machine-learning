import numpy as np
from functools import reduce

a = [1, 2, 3]

print(reduce(lambda x, y: x * y, a))

print(np.random.choice(100, 10))

print(len(np.array([[1, 2]])))

print(np.array(a)[::-1])

print(np.array(a)[:, np.newaxis])

print(np.sum(np.array([[1, 2, 3], [2, 3, 4]]), axis=1))
