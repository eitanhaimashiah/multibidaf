#!/bin/bash

allennlp fine-tune \
    -m /tmp/squad_serialization/model.tar.gz \
    -c training_config/multirc_config.json \
    -s /tmp/multirc_serialization \
    --include-package multibidaf
