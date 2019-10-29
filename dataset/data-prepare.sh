#!/bin/sh

# Copy image data from repository
GB_DATA_ROOT=~/ML-Study/_data/GB_images/

mkdir -p input/normal
find ${GB_DATA_ROOT}/normal -path "*/img-squared/*.png" | xargs -i cp {} input/normal

mkdir -p input/defects
find ${GB_DATA_ROOT}/defects -path "*/img-squared/*.png" | xargs -i cp {} input/defects

# Split dataset for train/test
split_folders input/ --ratio .8 .2
