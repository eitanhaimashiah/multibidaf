#!/bin/bash

python siamese/eval.py \
    --vocab_filepath /tmp/siamese/runs/1535977863/checkpoints/vocab \
    --model /tmp/siamese/runs/1535977863/checkpoints/graphpb.txt \
    --checkpoint_dir /tmp/siamese/runs/1535977863/checkpoints \
    --eval_filepath siamese/eval_file.tsv
