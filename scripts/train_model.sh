#!/bin/bash

allennlp fine-tune \
    -m bidaf_serialization/model.tar.gz \
    -c training_config/train_config.json \
    -s multibidaf_serialization \
    --include-package multibidaf
