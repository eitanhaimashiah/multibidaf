#!/bin/bash

./scripts/make_unified_vocab.sh ; ./scripts/pretrain_on_squad.sh ; ./scripts/train_on_multirc.sh
