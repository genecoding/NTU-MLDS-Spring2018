#!/bin/bash
python3 inference.py \
    --test_data $1 \
    --output_file $2 \
    --use_attention
