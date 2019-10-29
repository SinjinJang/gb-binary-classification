#!/usr/bin/env python
# coding: utf-8

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


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
train_set = train_datagen.flow_from_directory('../dataset/output/train',
                                              target_size=(128, 128),
                                              batch_size=128,
                                              class_mode='binary')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_set = validation_datagen.flow_from_directory('../dataset/output/val',
                                                        target_size=(128, 128),
                                                        batch_size=128,
                                                        class_mode='binary')
model.fit_generator(train_set,
                    steps_per_epoch=8000,
                    epochs=5,
                    validation_data=validation_set,
                    validation_steps=2000)

# Save model & weights
model_json = model.to_json()
with open('./model_cnn.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('./model_cnn.h5')
print('saved model, ready to go!')
