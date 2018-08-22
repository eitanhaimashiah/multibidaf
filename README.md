# Multi-BiDAF: Multiple Sentences Bi-directional Attention Flow
Multi Sentences Bi-directional Attention Flow (Multi-BiDAF) network is a model designed to fit the [BiDAF](https://github.com/allenai/bi-att-flow.git) model of Seo et al. (2016) for the [Multi-RC](https://github.com/CogComp/multirc.git) dataset. This implementation is built on the [AllenNLP](https://github.com/allenai/allennlp.git) library.

## Installation

To install Multi-BiDAF, start by cloning our git repository:

  ```bash
  git clone https://github.com/eitanhaimashiah/multibidaf.git
  ```

Create a Python 3.6 virtual environment, and install the necessary requirements by running:

  ```bash
  pip install -r requirements.txt
  ```

(The above is assuming CUDA 8 installed on a linux machine; use a different pytorch version as necessary.)

## Running Multi-BiDAF

Once you've installed Multi-BiDAF, you can run the command-line interface with `python -m multibidaf.run`.

```bash
$ python -m multibidaf.run
Run Multi-BiDAF

optional arguments:
  -h, --help    show this help message and exit

Commands:

    train       Train a model
    evaluate    Evaluate the specified model + dataset
    predict     Use a trained model to make predictions
    make-vocab  Create a vocabulary
    elmo        Use a trained model to make predictions
    fine-tune   Continue training a model on a new dataset
    dry-run     Create a vocabulary, compute dataset statistics and other
                training utilities.
    test-install
                Run the unit tests.
```


