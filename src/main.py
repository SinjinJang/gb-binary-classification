#!/usr/bin/env python
# coding: utf-8

""" Train and evaluate CNN models """

from argparse import ArgumentParser

from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from ModelCNN import TwoDepthCNN, BaseCNN


IMG_SIDE = 128
IMG_SHAPE = (IMG_SIDE, IMG_SIDE, 3)


def load_dataset(model: BaseCNN):
    print('\n<<< Loading dataset >>>')

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

    model.feed_data(train_set, val_set, test_set)


def do_train(model: BaseCNN):
    print('\n<<< Training >>>')

    ret = model.train()
    plt.plot(ret.history['acc'])
    plt.plot(ret.history['val_acc'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show(block=False)

    model.save()


def do_evaluate(model: BaseCNN):
    print('\n<<< Evaluation >>>')

    metrics_names, scores = model.evaluate()
    print(f'{metrics_names[0]}: {scores[0]*100:.2f}')
    print(f'{metrics_names[1]}: {scores[1]*100:.2f}%')


def parse_argument():
    parser = ArgumentParser(description='Command line argument description')
    parser.add_argument('-m', '--model-name', type=str,
                        required=True, help='specify model name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_argument()

    m = TwoDepthCNN(IMG_SHAPE)
    load_dataset(m)
    do_train(m)
    do_evaluate(m)

    _ = input("Press [enter] to exit.")
