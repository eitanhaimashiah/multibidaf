#!/bin/bash

allennlp train \
    multibidaf/tests/fixtures/squad_experiment.json \
    -s /tmp/test/squad_serialization \
    --include-package multibidaf
