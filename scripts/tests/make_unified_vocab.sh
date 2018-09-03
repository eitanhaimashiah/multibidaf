#!/bin/bash

allennlp make-vocab \
    multibidaf/tests/fixtures/squad_vocab_experiment.json \
    -s /tmp/test/vocabs/squad_vocab

allennlp make-vocab \
    multibidaf/tests/fixtures/unified_vocab_experiment.json \
    -s /tmp/test/vocabs/unified_vocab \
    --include-package multibidaf