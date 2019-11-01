#!/usr/bin/env python
# coding: utf-8

""" Basic CNN Model """

import abc
import datetime
from pathlib import Path

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json

IMG_SIDE = 128
IMG_SHAPE = (IMG_SIDE, IMG_SIDE, 3)


class BaseCNN:
    ''' Base class CNN model '''

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._train_set = None
        self._val_set = None
        self._test_set = None
        self._model: Sequential = self.make_model()

    @abc.abstractmethod
    def make_model(self):
        ''' abstract method that child class should implement their own model '''

    def feed_data(self, train_set, val_set, test_set):
        ''' Set data for train/validataion/test '''
        self._train_set = train_set
        self._val_set = val_set
        self._test_set = test_set

    def train(self):
        ''' Train model with dataset and hyperparameter '''
        self._model.summary()
        self._model.compile(optimizer='adam', loss='binary_crossentropy',
                            metrics=['accuracy'])
        return self._model.fit_generator(self._train_set,
                                         steps_per_epoch=500,
                                         epochs=10,
                                         validation_data=self._val_set,
                                         validation_steps=100)

    def evaluate(self):
        ''' Evaluate model '''
        scores = self._model.evaluate_generator(self._test_set,
                                                steps=5)
        return self._model.metrics_names, scores

    def save(self):
        ''' Save model '''
        now = datetime.datetime.today().strftime('%y%m%d-%H%M%S')
        model_path = Path('../model') / self.__class__.__name__ / now
        model_path.mkdir(parents=True, exist_ok=True)

        Path(model_path / 'model.json').write_text(self._model.to_json())
        self._model.save_weights(model_path / 'weights.h5')

    def load(self, model_path: Path):
        ''' Load model with weight '''
        model_text = (model_path / 'model.json').read_text()
        self._model = model_from_json(model_text)
        self._model.load_weights(model_path / 'weights.h5')


class TwoDepthCNN(BaseCNN):
    ''' 2 depth CNN model '''

    def __init__(self, input_shape):
        self.__input_shape = input_shape
        super().__init__()

    def make_model(self) -> Sequential:
        m = Sequential()
        m.add(Conv2D(32, (3, 3), input_shape=self.__input_shape, activation='relu'))
        m.add(MaxPooling2D(pool_size=(2, 2)))
        m.add(Conv2D(32, (3, 3), activation='relu'))
        m.add(MaxPooling2D(pool_size=(2, 2)))
        m.add(Flatten())
        m.add(Dense(units=128, activation='relu'))
        m.add(Dense(units=1, activation='sigmoid'))
        return m
