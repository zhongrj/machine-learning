import numpy as np
from functools import reduce


def reduce_product(array):
    return reduce(lambda x, y: x * y, array)


def isNaN(num):
    return num != num
