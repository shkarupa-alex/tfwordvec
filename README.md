# tfwordvec

Word2Vec and FastText models implemented with tf.keras.

Steroids:

- Regular or adaptive embedding storage
- Softmax or SampledSoftmax / NoiseContrastiveEstimation / AdaptiveSoftmax head
- Word / Ngram or Char / BPE / CNN (ELMo-style) word embeddings
- End-to-end tensorflow backedn training (including input preprocessing)
- Export as TFHub layer or word2vec binary format

## Configuration

To start using this package you should prepare config file.
Some examples can be found in `config` directory.
For full list of options see `tfwordvec/congif.py`.

## Data

This package use `*.txt.gz` files as input.

Models for `CHAR` unit can consume any raw text.
Models for other units expect each file to contain one sentence per line with words separated by space.

All additional preprocessing done inside the training/inference pipeline.

## Vocabulary

All models require two vocabularies: one for units and one for labels.
They can be estimated with `tfwordvec-vocab` console command.
Vocabularies are stored within dataset directory.

```bash
tfwordvec-vocab config/extfasttext.yaml data/dataset
```

## Training

Use `tfwordvec-train` to train model with your config, data and vocabularies.

```bash
tfwordvec-train config/extfasttext.yaml data/dataset models/extfasttext
```

## Hub

After model was trained, using `tfwordvec-hub` command you can export encoder part to `SavedModel` format for using
with `tensorflow_hub.KerasLayer`.

```bash
tfwordvec-hub config/extfasttext.yaml data/dataset models/extfasttext
```

## Export

Use `tfwordvec-export` to export `SavedModel` from hub to binary format.
You need to provide a vocabulary (`nlpvocab.Vocabulary` in pickle format) for export. For simplicity you can choose
label vocabulary used in training.

```bash
tfwordvec-export config/extfasttext.yaml models/extfasttext data/dataset/vocab_skipgram_ngram_label.pkl
```

## Evaluation

There is no in-package evaluation scripts.
To make evaluation use original word2vec evaluation scripts.
Some common datasets can be found in `datasets` directory.

More evaluation datasets can be found here:

- [rt, ae, ae2](https://russe.nlpub.org/task/)
- [assoc](https://github.com/dkulagin/kartaslov)
- [lrwc](https://github.com/natasha/corus#load_toloka_lrwc)

Also you may try more complex projects like:

- [Naeval](https://github.com/natasha/naeval)
- [DeepPavlovEval](https://github.com/deepmipt/deepPavlovEval)
- [SentEvalRu](https://github.com/comptechml/SentEvalRu)
- [SentEval](https://github.com/facebookresearch/SentEval)