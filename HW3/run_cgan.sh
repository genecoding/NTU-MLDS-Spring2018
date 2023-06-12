#!/bin/bash
python3 inference.py \
    --model CGAN \
    --load_path ./saved_model/cgan \
    --tag_file $1 \
    --save_imgname $2