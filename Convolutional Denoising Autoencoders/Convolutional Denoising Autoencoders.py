from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
import tensorflow.keras.models as models
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta

input = Input(shape=(28,28,1))

net = Conv2D(16, (3, 3), activation='relu', padding='same')(input)
net = MaxPooling2D(pool_size=(2, 2), padding='same')(net)
net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=(2, 2), padding='same')(net)
net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=(2, 2), padding='same')(net)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(net)

net = UpSampling2D(size=(2, 2))(encoded)
net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
net = UpSampling2D(size=(2, 2))(net)
net = Conv2D(16, (3, 3), activation='relu')(net)
net = UpSampling2D(size=(2, 2))(net)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(net)

autoencoder = Model(input, decoded)
autoencoder.compile(loss='binary_crossentropy', optimizer=Adadelta(lr=5.0/100, rho=0.95, epsilon=1e-07))

autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

import matplotlib.pyplot as plt

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # test image
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  
  # reconstructed image
  ax = plt.subplot(2, n, i + n + 1)
  plt.imshow(decoded_imgs[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
  ax = plt.subplot(1, n, i + 1)
  plt.imshow(x_test_noisy[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
import tensorflow.keras.models as models
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta

input = Input(shape=(28,28,1))

net = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
net = MaxPooling2D(pool_size=(2, 2), padding='same')(net)
net = Conv2D(32, (3, 3), activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=(2, 2), padding='same')(net)
encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(net)

net = UpSampling2D(size=(2, 2))(encoded)
net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
net = UpSampling2D(size=(2, 2))(net)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(net)

autoencoder = Model(input, decoded)
autoencoder.compile(loss='binary_crossentropy', optimizer=Adadelta(lr=5.0/100, rho=0.95, epsilon=1e-07))

autoencoder.fit(x_train_noisy, x_train, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test_noisy, x_test))

decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # x_test_noisy image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstructed image
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # x_test image
    ax = plt.subplot(3, n, i + n + n + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()