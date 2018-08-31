#!/bin/bash

allennlp train \
    multibidaf/tests/fixtures/squad_experiment.json \
    -s /tmp/squad_serialization \
    --include-package multibidaf
