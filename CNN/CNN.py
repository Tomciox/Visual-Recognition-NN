import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import numpy as np
data = np.load('mnist.npz')

# data = np.load('cifar10.npz')

Xtr,Ytr,Xte,Yte = data['Xtr'],data['Ytr'],data['Xte'],data['Yte']

# Xtr, Xte, input_shape = Xtr.reshape(Xtr.shape[0], 32, 32, 3), Xte.reshape(Xte.shape[0], 32, 32, 3), (32, 32, 3)

if K.image_data_format() == 'channels_first':
    Xtr, Xte, input_shape = Xtr.reshape(Xtr.shape[0], 1, 28, 28), Xte.reshape(Xte.shape[0], 1, 28, 28), (1, 28, 28)
else:
    Xtr, Xte, input_shape = Xtr.reshape(Xtr.shape[0], 28, 28, 1), Xte.reshape(Xte.shape[0], 28, 28, 1), (28, 28, 1)

# print(Xtr.shape[0], 'train samples and', Xte.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(Xtr, Ytr, batch_size=128, epochs=10, validation_data=(Xte, Yte))
score = model.evaluate(Xte, Yte, batch_size=128)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
