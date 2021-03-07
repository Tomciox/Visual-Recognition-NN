# Data is already preprocessed for VGG16, just load it.
import numpy as np
data = np.load('segmentation.npz')
train_x, train_y, test_x, test_y = data['train_x'], data['train_y'], data['test_x'], data['test_y']
del data

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.layers import Input, Conv2D, Dropout, Conv2DTranspose, Cropping2D
from keras.optimizers import SGD, Adam

nb_classes = 12

inputs = Input(shape=(360, 480, 3))
x = VGG16(weights='imagenet', include_top=False)(inputs)
x = Conv2D(2048, (7, 7), activation='relu', padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
x = Dropout(0.5)(x)

x = Conv2D(nb_classes, (1, 1), activation='relu', padding='same')(x)

x = Conv2DTranspose(filters=nb_classes, kernel_size=(64, 64), strides=(33, 32), padding='same', activation='sigmoid')(x)
x = Cropping2D(cropping=((0, 3), (0, 0)))(x)

model = Model(inputs=inputs, outputs=x)

model.layers[1].trainable = False
model.layers[2].trainable = False
model.layers[3].trainable = False


model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPOCHS = 10
STEPS_PER_EPOCH = 30

data_gen_args = dict(
          zoom_range=[0.8,1.0],
          horizontal_flip=True,
          width_shift_range=0.1, 
          height_shift_range=0.1,
          fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1

image_generator = image_datagen.flow(
          train_x,
          seed=seed,
          batch_size=3)

mask_generator = mask_datagen.flow(
          train_y,
          seed=seed,
          batch_size=3)

train_generator = zip(image_generator, mask_generator)

model.fit_generator(train_generator, validation_data=(test_x, test_y), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

model.layers[2].trainable = True
model.layers[3].trainable = True
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
model.fit_generator(train_generator, validation_data=(test_x, test_y), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

model.layers[1].trainable = True

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001, beta_1=0.5))
model.fit_generator(train_generator, validation_data=(test_x, test_y), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00005, beta_1=0.5))
model.fit_generator(train_generator, validation_data=(test_x, test_y), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.000025, beta_1=0.5))
model.fit_generator(train_generator, validation_data=(test_x, test_y), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00001, beta_1=0.5))
model.fit(train_x, train_y, epochs=2*EPOCHS, batch_size=3, validation_data=(test_x, test_y))

import matplotlib.pyplot as plt

pred = model.predict(np.expand_dims(test_x[0], axis=0))[0].argmax(axis=2)
plt.imshow(pred)

plt.imshow(test_y[0].argmax(axis=2))

model.save('./model.h5')

import tensorflow as tf
new_model = tf.keras.models.load_model('./model.h5')

pixels = {}

nb_classes = 12

for t in range(len(test_x)):
  pred = new_model.predict(np.expand_dims(test_x[t], axis=0))[0].argmax(axis=2)
  gt = test_y[t].argmax(axis=2)
  for i in range(len(pred)):
    for x, y in zip(pred[i], gt[i]):
      if (y, x) in pixels:
        pixels[y, x] += 1
      else:
        pixels[y, x] = 1
  
X = 0

for i in range(nb_classes-1):
  if (i, i) in pixels:
    X += pixels[i, i]

Y = 0

for i in range(nb_classes):
  ti = 0
  for j in range(nb_classes-1):
    if (i, j) in pixels:
      ti += pixels[i, j]
  Y += ti

print('pixel_accuracy = ', 100 * X / Y)