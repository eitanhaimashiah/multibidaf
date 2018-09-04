#!/bin/bash

allennlp predict \
    /tmp/multirc_serialization/model.tar.gz \
    data/multirc_dev.json \
    --output-file data/predicted_multirc_dev.json \
    --predictor machine-comprehension \
    --use-dataset-reader \
    --include-package multibidaf