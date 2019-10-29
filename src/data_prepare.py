#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import shutil

import cv2
import split_folders


IMG_SIZE = 128

DIR_IN_BASE = Path('/home/sinjin/ML-Study/_data/GB_images/')
DIR_OUT = Path('../dataset/')
DIR_TEMP = DIR_OUT / '_temp'


def resize_and_copy(cls_name):
    """ Resize and copy all images from data repository. """

    dir_in = DIR_IN_BASE / cls_name
    dir_temp = DIR_TEMP / cls_name
    dir_temp.mkdir(parents=True, exist_ok=True)

    images = [x for x in dir_in.rglob('*/img-squared/*.png')]
    for idx, each in enumerate(images):
        if idx % 10 == 0:
            print(f'{cls_name}: {idx} / {len(images)}')

        img = cv2.imread(str(each))
        if img is None:
            continue

        # Resize image and save it
        img_resized = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_AREA)
        dst_name = str(dir_temp / each.name)
        cv2.imwrite(dst_name, img_resized)


# Resize and copy images to temporary directory
resize_and_copy('normal')
resize_and_copy('defects')

# Split dataset for train/test
split_folders.ratio(str(DIR_TEMP), output=str(DIR_OUT), seed=1337, ratio=(.8, .2))

# Delete temporary directory
shutil.rmtree(DIR_TEMP)
