#!/bin/bash

allennlp make-vocab \
    training_config/squad_vocab_config.json \
    -s /tmp/vocabs/squad_vocab

allennlp make-vocab \
    training_config/unified_vocab_config.json \
    -s /tmp/vocabs/unified_vocab \
    --include-package multibidaf