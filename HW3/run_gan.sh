#!/bin/bash
python3 inference.py \
    --model DCGAN \
    --load_path ./saved_model/dcgan \
    --save_imgname $1