#!/usr/bin/env python
# coding: utf-8

""" Train CNN model with Batch Normal layer

Epoch 1/5
1000/1000 [==============================] - 246s 246ms/step - loss: 0.3204 - acc: 0.9529 - val_loss: 6.3815 - val_acc: 0.6041
Epoch 2/5
1000/1000 [==============================] - 240s 240ms/step - loss: 0.0626 - acc: 0.9878 - val_loss: 4.4357 - val_acc: 0.7122
Epoch 3/5
1000/1000 [==============================] - 241s 241ms/step - loss: 0.0201 - acc: 0.9954 - val_loss: 2.6905 - val_acc: 0.7755
Epoch 4/5
1000/1000 [==============================] - 240s 240ms/step - loss: 0.0192 - acc: 0.9950 - val_loss: 0.6394 - val_acc: 0.9408
Epoch 5/5
1000/1000 [==============================] - 240s 240ms/step - loss: 0.0179 - acc: 0.9958 - val_loss: 0.1321 - val_acc: 0.9796
"""

from pathlib import Path

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


MODEL_FILE_NAME = 'model_cnn-bn'

# CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(units=128, activation='relu'))
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
