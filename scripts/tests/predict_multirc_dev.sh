#!/bin/bash

allennlp predict \
    /tmp/test/multirc_serialization/model.tar.gz \
    multibidaf/tests/fixtures/multirc.json \
    --output-file multibidaf/tests/fixtures/predicted_multirc.json \
    --predictor machine-comprehension \
    --use-dataset-reader \
    --include-package multibidaf