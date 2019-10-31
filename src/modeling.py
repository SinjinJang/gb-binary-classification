#!/usr/bin/env python
# coding: utf-8

""" Train and evaluate CNN models """

from argparse import ArgumentParser
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

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_set = val_datagen.flow_from_directory('../dataset/val',
                                              target_size=(128, 128),
                                              batch_size=64,
                                              class_mode='binary')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('../dataset/test',
                                                target_size=(128, 128),
                                                batch_size=64,
                                                class_mode='binary')

    return train_set, val_set, test_set


def train_model(model: Sequential, train_set, val_set):
    """ Train model with dataset and hyperparameter """
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    ret = model.fit_generator(train_set,
                              steps_per_epoch=500,
                              epochs=10,
                              validation_data=val_set,
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


def model_3d_cnn():
    ''' 3 depth CNN '''
    m = Sequential()
    m.add(Conv2D(32, (3, 3), input_shape=IMG_SHAPE, activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(32, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(32, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(units=128, activation='relu'))
    m.add(Dense(units=1, activation='sigmoid'))

    return m


def model_4d_cnn():
    ''' 4 depth CNN '''
    m = Sequential()
    m.add(Conv2D(32, (3, 3), input_shape=IMG_SHAPE, activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(32, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(32, (3, 3), activation='relu'))
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


def parse_argument():
    parser = ArgumentParser(description='Command line argument description')
    parser.add_argument('-m', '--model-name', type=str,
                        required=True, help='specify model name')
    return parser.parse_args()


def init_model():
    MODEL_DICT['2d_cnn'] = model_2d_cnn
    MODEL_DICT['3d_cnn'] = model_3d_cnn
    MODEL_DICT['4d_cnn'] = model_4d_cnn
    MODEL_DICT['lenet-5'] = model_lenet5
    MODEL_DICT['inception-v3'] = model_inception_v3


def do_main(model_name):
    train_set, val_set, test_set = load_dataset()

    model = MODEL_DICT[model_name]()
    model.summary()

    train_model(model, train_set, val_set)
    save_model(model, model_name)
    evaluate_model(model, test_set)


if __name__ == '__main__':
    args = parse_argument()
    init_model()
    do_main(args.model_name)
