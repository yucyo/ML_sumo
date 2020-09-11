import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#model:

# (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
#
# train_images = train_images.reshape((60000, 28, 28, 1))
# test_images = test_images.reshape((10000, 28, 28, 1))
#
# # ピクセルの値を 0~1 の間に正規化
# train_images, test_images = train_images / 255.0, test_images / 255.0

#畳み込み基礎部分#relu()#softmax()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#Dense(隠れ層)レイヤー追加
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

x = [[[2,1,1],2],1,2]
print(len(x))
