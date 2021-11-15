# tfwordvec
Word2Vec and FastText models implemented with tf.keras.

## Configuration
To start using this package you should prepare config file.
Some examples can be found in `config` directory.
For full list of options see `tfwordvec/congif.py`.

## Data
This package use `*.txt.gz` files as input.

Models for `CHAR` unit can consume any raw text.
Models for other units expect each file to contain one sentence per line with words separated by space.

All additional preprocessing done inside the package.

## Vocabulary
All models require two vocabularies: one for units and one for labels.
They can be estimated with `tfwordvec-vocab` console command. 
Vocabularies are stored within dataset directory.

## Training
Use `tfwordvec-train` to train model with your config, data and vocabularies.

## Hub
After model was trained, using `tfwordvec-hub` command you can export encoder part to `SavedModel` format for using 
with `tensorflow_hub.KerasLayer`. 
See `tfwordvec/tests/test_hub.py` for example.

## Export
Use `tfwordvec-export` to export `SavedModel` from hub to binary format.

## Evaluation
There is no in-package evaluation scripts. 
To make evaluation use original word2vec evaluation scripts. 
Some common datasets can be found in `datasets` directory.

More evaluation datasets can be found here:
- [rt, ae, ae2](https://russe.nlpub.org/task/)
- [assoc](https://github.com/dkulagin/kartaslov)
- [lrwc](https://github.com/natasha/corus#load_toloka_lrwc)

Also you ma try more complex projects like:
-  [Naeval](https://github.com/natasha/naeval)
-  [DeepPavlovEval](https://github.com/deepmipt/deepPavlovEval)
-  [SentEvalRu](https://github.com/comptechml/SentEvalRu)
-  [SentEval](https://github.com/facebookresearch/SentEval)