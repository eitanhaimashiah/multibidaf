#!/bin/bash

allennlp fine-tune \
    -m pretrain_serialization/model.tar.gz \
    -c training_config/train_config.json \
    -s train_serialization \
    --include-package multibidaf
