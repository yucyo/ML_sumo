# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

idx_features = np.r_[1:30]
num_units_inside = 8

## Network definition
class MyJankenNetwork(chainer.Chain):
    def __init__(self):
        super(MyJankenNetwork, self).__init__(
            l1 = L.Linear(idx_features.shape[0], num_units_inside),
            l2 = L.Linear(num_units_inside, 3),
        )

    def __call__(self, x):
        h = self.l2(F.relu(self.l1(x)))
        return h

def transform(inputs):
    r = inputs[idx_features]
    label = int(inputs[-1])
    return r, label

def create():
    net = L.Classifier(MyJankenNetwork())
    return net
