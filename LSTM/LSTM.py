from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_val, y_val = x_train[-10000:], y_train[-10000:]
x_train, y_train = x_train[:-10000], y_train[:-10000]

x_val = x_val / 255.0
x_train = x_train / 255.0
x_test = x_test / 255.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape, Conv2D, Flatten

model = Sequential()

model.add(Reshape(target_shape=(28,28, 1)))
model.add(Conv2D(filters=1, kernel_size=3, padding='same'))
model.add(Reshape(target_shape=(28,28)))

model.add(LSTM(128, input_shape=(28,28), activation='relu', return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.003, decay=0.00005), metrics=['acc'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

model.evaluate(x_test, y_test)

from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

def get_gradcam_heatmap(model, image):
  image = image.reshape(1, 28, 28)

  gradient_model = Model(inputs=[model.inputs], 
                        outputs=[model.get_layer("conv2d").output,
                                  model.output])

  with tf.GradientTape() as tape:
    (output, predictions) = gradient_model(image)

    loss = predictions[0, tf.argmax(predictions[0])]

    # tape.watch(output)

    grads = tape.gradient(loss, output)[0]

    output = output.numpy()[0]

    heatmap = np.mean(output * grads, axis=2)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

import matplotlib.pyplot as plt

plt.imshow(x_test[0])
plt.plot()

heatmap = get_gradcam_heatmap(model, x_test[0])
plt.matshow(heatmap)
plt.show()
