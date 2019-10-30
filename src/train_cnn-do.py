#!/usr/bin/env python
# coding: utf-8

""" Train CNN model with Dropout layer

Epoch 1/5
1000/1000 [==============================] - 243s 243ms/step - loss: 0.2206 - acc: 0.9090 - val_loss: 0.1233 - val_acc: 0.9531
Epoch 2/5
1000/1000 [==============================] - 243s 243ms/step - loss: 0.0680 - acc: 0.9743 - val_loss: 0.0992 - val_acc: 0.9653
Epoch 3/5
1000/1000 [==============================] - 243s 243ms/step - loss: 0.0353 - acc: 0.9869 - val_loss: 0.0980 - val_acc: 0.9714
Epoch 4/5
1000/1000 [==============================] - 243s 243ms/step - loss: 0.0254 - acc: 0.9907 - val_loss: 0.1106 - val_acc: 0.9776
Epoch 5/5
1000/1000 [==============================] - 240s 240ms/step - loss: 0.0160 - acc: 0.9946 - val_loss: 0.1140 - val_acc: 0.9694
"""

from pathlib import Path

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator


MODEL_FILE_NAME = 'model_cnn-do'

DROPOUT_RATE = 0.25

# CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(DROPOUT_RATE))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(DROPOUT_RATE))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

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
