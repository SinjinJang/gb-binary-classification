#!/bin/sh
 
# Resize and copy image data from repository
GB_DATA_ROOT=~/ML-Study/_data/GB_images/
TEMP_DIR=_temp

for CLASS in normal defects; do
    mkdir -p ${TEMP_DIR}/${CLASS}
    for EACH in `find ${GB_DATA_ROOT}/${CLASS} -path "*/img-squared/*.png"`; do
        echo ${EACH}
        convert ${EACH} -resize 128x128\> ${TEMP_DIR}/${CLASS}/`basename ${EACH}`
    done
done

# Split dataset for train/test
split_folders ${TEMP_DIR} --output . --ratio .8 .2

rm -r ${TEMP_DIR}
