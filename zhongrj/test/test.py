import numpy as np
from functools import reduce
import zhongrj.utils.math_util as math_util

a = [1, 2, 3]

print(math_util.reduce_product(a))

print(np.random.choice(100, 10))

print(len(np.array([[1, 2]])))

print(np.array(a)[::-1])

print(np.array(a)[:, np.newaxis])

print(np.sum(np.array([[1, 2, 3], [2, 3, 4]]), axis=1))

print(8 / 2 ** 2)

for i in reversed(range(1, 10)):
    print(i)

print(np.random.normal(0, 1, size=(50, 100)).astype(np.float32))

# print(np.array([-1e1000]) + np.array([1e1000]))

print(np.NaN == np.NaN)

for i in range(1, 1):
    print("1231231234134")

b = object()
c = {b: 1}
print(c[b])

for i, j in enumerate(reversed([10, 20, 30])):
    print(i, ' --- ', j)

print('{}'.format(123))

print([1, 2, 3, 4][:-1])

print([1, 2] + list())

lambda_test = lambda o, **kwargs: o
print(lambda_test(123, test=1123, test1=123))

print(np.concatenate([np.array(a), np.array(a)]))

print(np.hstack(([1], [2])))

print(np.hstack((1, True, False)))

d = np.arange(9).reshape([3, 3])
for row in d:
    row[0] = 100
print(d)

e = f = 1
print(e)
print(f)

g = np.arange(0, 9).reshape([3, 3]).astype(np.float32)
h = g[:, 2]
g[:, 2] = (h - h.mean()) / h.std()
print(g)

print(1 > -np.inf)

for i in range(10):
    print(np.random.choice(2, p=np.array([0.1, 0.9])))

basic = np.linspace(-1, 1, 10)
x = np.tile(basic[:, np.newaxis], (1, len(basic))).reshape([-1])
y = np.tile(basic, (len(basic),))
print(np.dstack((x, y)).squeeze())

l = np.array([1, 2, 3, 4])
# print(np.random.choice(3, 2))
print(l[np.array([1, 2])])

for i, j in zip([1, 2, 3], [2, 3]):
    print(i, '---', j)

print(np.tile([1, 0], (10,)))

o = {
    'temp': np.array([1, 2])
}
print(o)

import math

n = 10000
print(n * math.tan(math.pi / n))

# 修改文件名
# import os
# import sys
#
# path = "D:\study\generate_anime_face"
# for (path, dirs, files) in os.walk(path):
#     for filename in files:
#         if filename.find('00.') != -1 or filename.find('25.') != -1 or filename.find('50.') != -1 or filename.find(
#                 '75.') != -1:
#             # print(filename)
#             os.rename(path + "\\" + filename, path + "\\" + filename.replace('sample', 'aaaaaa'))
