#!/bin/bash

./scripts/tests/make_unified_vocab.sh ; ./scripts/tests/pretrain_on_squad.sh ; ./scripts/tests/train_on_multirc.sh
