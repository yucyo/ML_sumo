# -*- coding: utf-8 -*-

import numpy as np

def read():
    # read data
    with open("input.txt", "r") as fr:
        data = [[int(val) for val in l.split(None)] for l in fr]
        return np.array(data, dtype=np.float32)
