#!/bin/bash

allennlp train \
    training_config/pretrain_config.json \
    -s bidaf_serialization2 \
    --include-package multibidaf
