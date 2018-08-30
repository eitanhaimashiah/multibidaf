#!/bin/bash

allennlp train \
    training_config/pretrain_config.json \
    -s pretrain_serialization \
    --include-package multibidaf
