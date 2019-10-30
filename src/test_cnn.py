#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator


MODEL_FILE_NAME = 'model_cnn'

model = model_from_json(Path(f'{MODEL_FILE_NAME}.json').read_text())
model.load_weights(f'{MODEL_FILE_NAME}.h5')
print("Loaded model from disk")

model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('../dataset/test',
                                            target_size=(128, 128),
                                            batch_size=64,
                                            class_mode='binary')

scores = model.evaluate_generator(test_set, steps=5)
print(test_set.class_indices)
print(f'{model.metrics_names[0]}: {scores[0]*100:.2f}')
print(f'{model.metrics_names[1]}: {scores[1]*100:.2f}%')
