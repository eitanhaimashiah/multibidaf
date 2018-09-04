# Multi-BiDAF: Multiple Sentences Bi-directional Attention Flow
Multi Sentences Bi-directional Attention Flow (Multi-BiDAF) network is a model designed to fit the [BiDAF](https://github.com/allenai/bi-att-flow.git) model of Seo et al. (2017) for the [Multi-RC](https://github.com/CogComp/multirc.git) dataset. This implementation is built on the [AllenNLP](https://github.com/allenai/allennlp.git) library.

## Installation

To install Multi-BiDAF, start by cloning our git repository:

  ```bash
  $ git clone https://github.com/eitanhaimashiah/multibidaf.git
  ```

Create a Python 3.6 virtual environment, and install the necessary requirements by running:

  ```bash
  $ ./scripts/install_requirements.sh
  ```

(The above is assuming CUDA 9 installed on a linux machine; use a different pytorch version as necessary.)

## Training Multi-BiDAF

Once you've installed Multi-BiDAF, you can train our model fully by running:

```bash
$ ./scripts/train_fully.sh
```
When you run this it will compute an unified vocabulary for the SQuAD and MultiRC datasets, 
pretrain the Multi-BiDAF model on SQuAD, and eventually train the model on MultiRC. 
Each of these tasks can be accomplished by running separate scripts 
(`scripts/make_unified_vocab.sh`, `scripts/pretrain_on_squad.sh`, `scripts/train_on_multirc.sh`, respectively).

Moreover, you can create a prediction file (adapted to the official MultiRC evaluation script) of the 
development set by running:
  ```bash
  $ ./scripts/predict_multirc_dev.sh
  ```
