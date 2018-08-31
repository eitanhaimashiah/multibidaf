#!/bin/bash

allennlp fine-tune \
    -m /tmp/squad_serialization/model.tar.gz \
    -c multibidaf/tests/fixtures/multirc_experiment.json \
    -s /tmp/multirc_serialization \
    --include-package multibidaf
