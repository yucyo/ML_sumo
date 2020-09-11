#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer.datasets import TransformDataset
import mymodel
import mydata

## read data
data_orig = mydata.read()

win_total = 0
draw_total = 0
lose_total = 0

for year in xrange(1996, 2019):
    ## network setup
    net = mymodel.create()
    chainer.serializers.load_npz("models/{0}.npz".format(year), net)

    # 前年までのデータで学習したモデルで、1年分のデータを予測
    data = data_orig[data_orig[:, 0] == year]
    inputs = TransformDataset(data, mymodel.transform)
    win = 0
    draw = 0
    lose = 0
    for testcase in inputs:
        detected = net.predictor(testcase[0].reshape((1,-1))).data.argmax(axis=1)[0]
        # 相手が最も出す確率の高い手に勝つように出す
        mychoice = (detected + 2) % 3 # 0: G, 1: C, 2: P
        schoice = testcase[1]
        if (mychoice + 3 - schoice) % 3 == 0:
            draw += 1
        elif (mychoice + 3 - schoice) % 3 == 1:
            lose += 1
        else:
            win += 1

    win_total += win
    draw_total += draw
    lose_total += lose
    print "{0}    Win: {1}, Draw: {2}, Lose: {3} / Avg: {4:.2f}%".format(year, win, draw, lose, float(win)/(win+lose)*100)

print "======"
print "<TOTAL> Win: {0}, Draw: {1}, Lose: {2} / Avg: {3:.2f}%".format(win_total, draw_total, lose_total, float(win_total)/(win_total+lose_total)*100)
