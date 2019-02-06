## intrdouction to CNN using Keras
# https://notebooks.azure.com/anon-manzxw/projects/NeuralNetworks/html/Introduction%20to%20Convolution%20Neural%20Networks.ipynb
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical

import matplotlib.pyplot as plt
## matplotlib inline
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
//Pre - processing
//Our MNIST images only have a depth of 1, but we must explicitly declare that
num_classes = 10
epochs = 3

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

## The first parameter is the number of filters

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size = (5, 5), input_shape = (28, 28, 1), padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D())
cnn.add(Conv2D(64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D())
cnn.add(Flatten())
cnn.add(Dense(1024, activation = 'relu'))
cnn.add(Dense(10, activation = 'softmax'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(cnn.summary())

history_cnn = cnn.fit(X_train, y_train, epochs = 5, verbose = 1, validation_data = (X_train, y_train))

plt.plot(history_cnn.history['acc'])
plt.plot(history_cnn.history['val_acc'])

## The accuracy of the model¶
plt.plot(history_cnn.history['acc'])
plt.plot(history_cnn.history['val_acc'])

## Prediction
model.predict()