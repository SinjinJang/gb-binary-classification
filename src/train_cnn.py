#!/usr/bin/env python
# coding: utf-8

""" Train pure CNN model

Epoch 1/5
1000/1000 [==============================] - 252s 252ms/step - loss: 0.1905 - acc: 0.9198 - val_loss: 0.1045 - val_acc: 0.9694
Epoch 2/5
1000/1000 [==============================] - 246s 246ms/step - loss: 0.0450 - acc: 0.9831 - val_loss: 0.1400 - val_acc: 0.9673
Epoch 3/5
1000/1000 [==============================] - 240s 240ms/step - loss: 0.0193 - acc: 0.9933 - val_loss: 0.1848 - val_acc: 0.9612
Epoch 4/5
1000/1000 [==============================] - 237s 237ms/step - loss: 0.0139 - acc: 0.9952 - val_loss: 0.1759 - val_acc: 0.9653
Epoch 5/5
1000/1000 [==============================] - 240s 240ms/step - loss: 0.0119 - acc: 0.9956 - val_loss: 0.1511 - val_acc: 0.9673
"""

from pathlib import Path

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


MODEL_FILE_NAME = 'model_cnn'

# CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_set = train_datagen.flow_from_directory('../dataset/train',
                                              target_size=(128, 128),
                                              batch_size=64,
                                              class_mode='binary')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_set = validation_datagen.flow_from_directory('../dataset/val',
                                                        target_size=(128, 128),
                                                        batch_size=64,
                                                        class_mode='binary')
model.fit_generator(train_set,
                    steps_per_epoch=1000,
                    epochs=5,
                    validation_data=validation_set,
                    validation_steps=1000)

# Save model & weights
Path(f'{MODEL_FILE_NAME}.json').write_text(model.to_json())
model.save_weights(f'{MODEL_FILE_NAME}.h5')

print('saved model, ready to go!')
