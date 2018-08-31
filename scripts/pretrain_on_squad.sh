#!/bin/bash

allennlp train \
    training_config/squad_config.json \
    -s /tmp/squad_serialization \
    --include-package multibidaf
