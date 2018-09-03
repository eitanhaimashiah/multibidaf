#!/bin/bash

allennlp fine-tune \
    -m /tmp/test/squad_serialization/model.tar.gz \
    -c multibidaf/tests/fixtures/multirc_experiment.json \
    -s /tmp/test/multirc_serialization \
    --include-package multibidaf
