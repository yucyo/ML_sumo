# -*- coding:utf-8 -*-
#python 3.7, conda 4.7.12, tensorflow 2.0.0b1, keras 2.3.1, rl 0.4.2, rl-2 1.0.3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Activation,Dense
from keras.models import Sequential
from keras.recurrent import LSTM

class Prediction :

  def __init__(self):
    self.length_of_sequences = 10
    self.in_out_neurons = 1
    self.hidden_neurons = 300


  def load_data(self, data, n_prev=10):
    X, Y = [], []
    for i in range(len(data) - n_prev):
      X.append(data.iloc[i:(i+n_prev)].as_matrix())
      Y.append(data.iloc[i+n_prev].as_matrix())
    retX = numpy.array(X)
    retY = numpy.array(Y)
    return retX, retY


  def create_model(self) :
    model = Sequential()
    model.add(LSTM(self.hidden_neurons, \
              batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), \
              return_sequences=False))
    model.add(Dense(self.in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mape", optimizer="adam")
    return model


  def train(self, X_train, y_train) :
    model = self.create_model()
    # 学習
    model.fit(X_train, y_train, batch_size=10, nb_epoch=100)
    return model


if __name__ == "__main__":

  prediction = Prediction()

  # データ準備
  data = None
  for year in range(2007, 2017):
    data_ = pd.read_csv('csv/indices_I101_1d_' + str(year) +  '.csv')
    data = data_ if (data is None) else pd.concat([data, data_])
  data.columns = ['date', 'open', 'high', 'low', 'close']
  data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
  # 終値のデータを標準化
  data['close'] = preprocessing.scale(data['close'])
  data = data.sort_values(by='date')
  data = data.reset_index(drop=True)
  data = data.loc[:, ['date', 'close']]

  # 2割をテストデータへ
  split_pos = int(len(data) * 0.8)
  x_train, y_train = prediction.load_data(data[['close']].iloc[0:split_pos], prediction.length_of_sequences)
  x_test,  y_test  = prediction.load_data(data[['close']].iloc[split_pos:], prediction.length_of_sequences)

  model = prediction.train(x_train, y_train)

  predicted = model.predict(x_test)
  result = pd.DataFrame(predicted)
  result.columns = ['predict']
  result['actual'] = y_test
  result.plot()
  plt.show()
