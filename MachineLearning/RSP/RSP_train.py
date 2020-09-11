#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import chainer
from chainer.datasets import TransformDataset
from chainer.training import extensions, triggers
import mymodel
import mydata

# 再現性のために乱数シードを固定
# https://qiita.com/mitmul/items/1e35fba085eb07a92560
def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

## Main
gpu_id = -1
max_epoch = 50

## read data
data_orig = mydata.read()

## make output directory
try:
    os.makedirs("models")
except OSError:
    pass

for year in range(1992, 2019):
    print ("Year: {0}".format(year))
    reset_seed(0)

    # 前年までのデータで学習
    data = data_orig[data_orig[:, 0] < year]
    inputs = TransformDataset(data, mymodel.transform)
    batchsize = max(len(inputs) / 128, 1)
    print ("Data size: {0}".format(len(inputs)))
    print ("Batch size: {0}".format(batchsize))

    train, valid = chainer.datasets.split_dataset_random(inputs, len(inputs)*9/10, seed=0)
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)

    # network setup
    net = mymodel.create()

    # trainer setup
    optimizer = chainer.optimizers.Adam().setup(net)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    # early stopping
    # https://qiita.com/klis/items/7865d9e8e757f16bc39c
    stop_trigger = triggers.EarlyStoppingTrigger(monitor='val/main/loss', max_trigger=(max_epoch, 'epoch'))
    trainer = chainer.training.Trainer(updater, stop_trigger)
    trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ExponentialShift("alpha", 0.9999))

    ## Let's train!
    trainer.run()

    ## save model
    chainer.serializers.save_npz("models/{0}.npz".format(year), net)
