from tensorflow.keras.preprocessing.image import ImageDataGenerator

imgs_train_dir = 'dogs-vs-cats-2k/train/'
imgs_val_dir = 'dogs-vs-cats-2k/val/'

# Rescale images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(imgs_train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(imgs_val_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import RMSprop

model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=1e-4), metrics=['acc'])

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator( train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50 )

model.save('dogs_vs_cats_small_v1.h5')

print(history.history)

import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_loss"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(30)

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

datagen = ImageDataGenerator( rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest' )

from tensorflow.keras.preprocessing import image

f = 'dogs-vs-cats-2k/train/dog/dog.70.jpg'
img = image.load_img(f, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
  plt.figure(i)
  imgplot = plt.imshow(image.array_to_img(batch[0]))
  i += 1
  if i % 4 == 0:
    break
plt.show()

model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)))  # conv 3x3 32 filters + relu
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling 2x2
model.add(Conv2D(64, (3, 3), activation='relu'))  # conv 3x3 64 filters + relu
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling 2x2
model.add(Conv2D(128, (3, 3), activation='relu'))  # conv 3x3 128 filters + relu
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling 2x2
model.add(Conv2D(128, (3, 3), activation='relu'))  # conv 3x3 128 filters + relu
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling 2x2
model.add(Flatten())
model.add(layers.Dropout(0.5))
model.add(Dense(512, activation='relu'))  # FC 512 + relu
model.add(Dense(1, activation='sigmoid'))  # FC + sigmoid
model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=1e-4), metrics=['acc'])  # RMSprop 1e-4 + accuracy metric

train_datagen = ImageDataGenerator( rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True )
test_datagen = ImageDataGenerator( rescale=1./255 )

train_generator = train_datagen.flow_from_directory( imgs_train_dir, target_size=(150, 150), batch_size=20, class_mode='binary' )
validation_generator = test_datagen.flow_from_directory( imgs_val_dir, target_size=(150, 150), batch_size=20, class_mode='binary' )

history = model.fit_generator( train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50 )

model.save('dogs_vs_cats_small_v2.h5')

acc = history.history["acc"]
val_acc = history.history["val_loss"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(30)

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

