from tensorflow.keras.preprocessing.image import ImageDataGenerator

imgs_train_dir = 'dogs-vs-cats-2k/train'
imgs_val_dir = 'dogs-vs-cats-2k/val'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(imgs_train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(imgs_val_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

model = models.Sequential()
model.add(vgg16)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

vgg16.trainable = False

from tensorflow.keras import optimizers

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-5), metrics=['acc'])  # loss, optimizer=RMSprop, metrics=acc
history = model.fit_generator( train_generator, steps_per_epoch=100, epochs=20, validation_data=validation_generator, validation_steps=50 )

import matplotlib.pyplot as plt

# get from variable "history"
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = [i for i in range(20)]

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()