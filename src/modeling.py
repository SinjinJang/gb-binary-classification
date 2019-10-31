#!/usr/bin/env python
# coding: utf-8

""" Train and evaluate CNN models """

import datetime
from pathlib import Path

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3

import matplotlib.pyplot as plt


MODEL_DICT = {}

IMG_SIDE = 128
IMG_SHAPE = (IMG_SIDE, IMG_SIDE, 3)


def load_dataset():
    """ Load train, validation, test dataset """
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
                                                            target_size=(
                                                                128, 128),
                                                            batch_size=64,
                                                            class_mode='binary')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('../dataset/test',
                                                target_size=(128, 128),
                                                batch_size=64,
                                                class_mode='binary')

    return train_set, validation_set, test_set


def train_model(model: Sequential, train_set, validation_set):
    """ Train model with dataset and hyperparameter """
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    ret = model.fit_generator(train_set,
                              steps_per_epoch=500,
                              epochs=10,
                              validation_data=validation_set,
                              validation_steps=100)

    plt.plot(ret.history['acc'])
    plt.plot(ret.history['val_acc'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()


def evaluate_model(model, test_set):
    """ Evaluate model """
    scores = model.evaluate_generator(test_set, steps=5)

    print(test_set.class_indices)
    print(f'{model.metrics_names[0]}: {scores[0]*100:.2f}')
    print(f'{model.metrics_names[1]}: {scores[1]*100:.2f}%')


def save_model(model, key):
    now = datetime.datetime.today().strftime('%y%m%d-%H%M%S')
    model_path = Path('../model') / key / now
    model_path.mkdir(parents=True, exist_ok=True)

    Path(model_path / 'model.json').write_text(model.to_json())
    model.save_weights(model_path / 'weights.h5')


def model_2d_cnn():
    ''' 2 depth CNN '''
    m = Sequential()
    m.add(Conv2D(32, (3, 3), input_shape=IMG_SHAPE, activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(32, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(units=128, activation='relu'))
    m.add(Dense(units=1, activation='sigmoid'))

    return m


def model_lenet5():
    """ LeNet-5 """
    m = Sequential()
    m.add(Conv2D(filters=6, kernel_size=(3, 3),
                 activation='relu', input_shape=IMG_SHAPE))
    m.add(AveragePooling2D())
    m.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    m.add(AveragePooling2D())
    m.add(Flatten())
    m.add(Dense(units=120, activation='relu'))
    m.add(Dense(units=84, activation='relu'))
    m.add(Dense(units=1, activation='sigmoid'))

    return m


def model_inception_v3():
    """ GoogLeNet v3 """
    input_tensor = Input(shape=IMG_SHAPE)
    return InceptionV3(input_tensor=input_tensor, weights=None, include_top=True, classes=1)


def initmodel():
    MODEL_DICT['2d_cnn'] = model_2d_cnn
    MODEL_DICT['lenet-5'] = model_lenet5
    MODEL_DICT['inception-v3'] = model_inception_v3


def main():
    train_set, val_set, test_set = load_dataset()

    k = 'lenet-5'
    m = MODEL_DICT[k]()
    m.summary()

    train_model(m, train_set, val_set)
    save_model(m, k)
    evaluate_model(m, test_set)


if __name__ == '__main__':
    initmodel()
    main()
