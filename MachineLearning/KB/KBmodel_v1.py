# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
from keras.layers import Activation,Dense,Dropout
from keras.models import Sequential
from keras.recurrent import LSTM

# データ取得
data_file='KBdata.tsv'
res_file='small_neural'
df = res[['time'],'openASK']
df.columns = ['time','open']

### LSTM_window設定
window_len = 20

split_date = 'yyyy/mm/dd 00:00:00'
train, test = df[df['time'] < split_date],df[df['time'] >= split_date]
latest = test[:window_len]
del train['time']
del test['time']
del latest['time']
length = len(test) - window_len

### LSTM入力用への変換処理関数
def data_maker(data):
    data_lstm_in=[]
    if len(data) == window_len:
        temp = data[:window_len].copy()
        temp = temp / temp.iloc[0] - 1
        data_lstm_in.append(temp)
    for i in range(len(data) - window_len):
        temp = data[i : (i + window_len)].copy()
        temp = temp / temp.iloc[0] - 1
        data_lstm_in.append(temp)
    return data_lstm_in

train_lstm_in = data_maker(train)
lstm_train_out = (train['open'][window_len:].values / train['open'][:-window_len].values)-1
test_lstm_in = data_maker(test)
lstm_test_out = (test['open'][window_len:].values / test['open'][:-window_len].values)-1
latest_lstm_in = data_maker(latest)



# データ加工


# モデル構築


# 学習と予測
