import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Model

from keras.datasets.fashion_mnist import load_data as load_data_fmnist
from keras.datasets.mnist import load_data as load_data_mnist
from keras.utils import to_categorical

from resnet import ResNet

# Variables
img_size = 28
num_classes = 10

# Load dataset
# data_train, data_test = load_data_mnist()
data_train, data_test = load_data_fmnist()

(X_train, y_train), (X_test, y_test) = data_train, data_test
x_train = X_train.reshape(X_train.shape[0],img_size,img_size,1)
x_test = X_test.reshape(X_test.shape[0],img_size,img_size,1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test,num_classes)

model = ResNet().model()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
model.summary()

epochs = 2
batch_size = 100

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.15,
              shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

if __name__ == '__main__':
	print('all done!')
