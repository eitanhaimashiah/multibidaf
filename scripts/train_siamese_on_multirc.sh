#!/bin/bash

python siamese/train.py \
    --training_files siamese/mini_person_match1.tsv siamese/mini_person_match2.tsv
