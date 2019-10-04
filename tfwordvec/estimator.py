from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nlpvocab
import six
import tensorflow as tf
from tfwordvec.word2vec.model import _UNIQUE_LABEL, build_model_fn

_TRAIN_OPTIMIZER = 'Adam'
_LEARNING_RATE = 0.05


class VectorEstimator(tf.estimator.Estimator):
    def __init__(self, freq_vocab, min_freq, embed_size, loss_func, sampled_count,
                 train_opt=_TRAIN_OPTIMIZER, learn_rate=_LEARNING_RATE, sparse_comb='mean',
                 model_dir=None, config=None, warm_start_from=None):

        if not isinstance(freq_vocab, nlpvocab.Vocabulary):
            raise ValueError('Frequency vocabulary should be an instance of `nlpvocab.Vocabulary`. '
                             'Given type: {}'.format(type(freq_vocab)))
        unk_vocab = freq_vocab.trim(min_freq)
        freq_vocab[_UNIQUE_LABEL] += six.itervalues(unk_vocab)
        self.labels_vocab = freq_vocab.tokens()

        self.embed_size = embed_size
        self.loss_func = loss_func
        self.sampled_count = sampled_count
        self.train_opt = train_opt
        self.learn_rate = learn_rate
        self.sparse_comb = sparse_comb

        super(VectorEstimator, self).__init__(
            model_fn=self._model_fn,
            model_dir=model_dir,
            config=config,
            params=None,
            warm_start_from=warm_start_from
        )

    def _model_fn(self, features, labels, mode, params, config):
        return build_model_fn(
            features=features,
            labels=labels,
            mode=mode,
            params=params,
            config=config,
            labels_vocab=self.labels_vocab,
            embed_size=self.embed_size,
            sparse_comb=self.sparse_comb,
            loss_func=self.loss_func,
            sampled_count=self.sampled_count,
            train_opt=self.train_opt,
            learn_rate=self.learn_rate,
        )
