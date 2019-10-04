from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import six
import tensorflow as tf
from tensorflow.contrib.estimator import clip_gradients_by_norm
from tensorflow.python.estimator.canned.optimizers import get_optimizer_instance

_UNIQUE_LABEL = '<UNK>'
_CLIP_NORM = 5.0


class VectorModel(object):
    CHAR_SKIPGRAM = 'CHAR_SKIPGRAM'
    CHAR_CBOW = 'CHAR_CBOW'
    WORD_SKIPGRAM = 'WORD_SKIPGRAM'
    WORD_CBOW = 'WORD_CBOW'
    NGRAM_SKIPGRAM = 'NGRAM_SKIPGRAM'  # FastText

    @classmethod
    def all(cls):
        return cls.CHAR_SKIPGRAM, cls.CHAR_CBOW, cls.WORD_SKIPGRAM, cls.WORD_CBOW, cls.NGRAM_SKIPGRAM

    @classmethod
    def validate(cls, key):
        if not isinstance(key, six.string_types) or key not in cls.all():
            raise ValueError('Invalid vector model {}'.format(key))


class LossFunction(object):
    NOISE_CONTRASTIVE_ESTIMATION = 'NOISE_CONTRASTIVE_ESTIMATION'
    SAMPLED_SOFTMAX = 'SAMPLED_SOFTMAX'

    @classmethod
    def all(cls):
        return cls.NOISE_CONTRASTIVE_ESTIMATION, cls.SAMPLED_SOFTMAX

    @classmethod
    def validate(cls, key):
        if not isinstance(key, six.string_types) or key not in cls.all():
            raise ValueError('Invalid loss function {}'.format(key))


def build_model_fn(labels_vocab, embed_size, sparse_comb, loss_func, sampled_count, train_opt, learn_rate,
                   features, labels, mode, params, config):
    del params, config

    if _UNIQUE_LABEL not in labels_vocab:
        raise ValueError('Labels vocabulary should contain "unique" ({}) label'.format(_UNIQUE_LABEL))
    vocab_size = len(labels_vocab)
    unique_id = labels_vocab.index(_UNIQUE_LABEL)

    if not isinstance(features, dict):
        raise ValueError('Features should be a dict of `Tensor`s. Given type: {}'.format(type(features)))
    if 'source' not in features:
        raise ValueError('Features should contain `source` key. Given keys: {}'.format(list(features.keys())))
    if not isinstance(features['source'], tf.Tensor) and not isinstance(features['source'], tf.SparseTensor):
        raise ValueError(
            'Feature `source` should be a `Tensor` or a `SpaseTensor`. Given type: {}'.format(type(features['source'])))

    partitioner = tf.variable_axis_size_partitioner(
        max_shard_bytes=2 ** 30  # 1Gb
    )

    with tf.variable_scope('model', values=tuple(six.itervalues(features)), partitioner=partitioner):
        # with tf.device('cpu:0'):  # TODO
        embedding_weights = tf.get_variable(
            name='embedding_weights',
            shape=[vocab_size, embed_size],
            initializer=tf.random_uniform_initializer(
                minval=-0.5 / embed_size,
                maxval=0.5 / embed_size
            )
        )

        lookup_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=labels_vocab,
            default_value=unique_id
        )

        source_ids = lookup_table.lookup(features['source'])
        if isinstance(features, tf.SparseTensor):
            source_embeddings = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=embedding_weights,
                sparse_ids=source_ids,
                combiner=sparse_comb,
                default_id=unique_id,
                partition_strategy='div',
            )
        else:
            source_embeddings = tf.nn.embedding_lookup(
                params=embedding_weights,
                ids=source_ids,
                partition_strategy='div',
            )

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'vectors': source_embeddings}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        softmax_weights = tf.get_variable(
            name='softmax_weights',
            shape=[vocab_size, embed_size],
            initializer=tf.truncated_normal_initializer(
                stddev=1.0 / math.sqrt(embed_size)
            )
        )
        softmax_biases = tf.get_variable(
            name='softmax_biases',
            shape=[vocab_size],
            initializer=tf.zeros_initializer()
        )

        if mode == tf.estimator.ModeKeys.EVAL:
            source_logits = tf.nn.xw_plus_b(source_embeddings, tf.transpose(softmax_weights), softmax_biases)
            loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(labels, vocab_size),
                logits=source_logits
            )
            return tf.estimator.EstimatorSpec(mode, loss=loss_op, eval_metric_ops={})

        label_ids = tf.expand_dims(
            lookup_table.lookup(labels),
            axis=-1
        )

        LossFunction.validate(loss_func)
        if LossFunction.NOISE_CONTRASTIVE_ESTIMATION == loss_func:
            loss_op = tf.reduce_mean(tf.nn.nce_loss(
                weights=softmax_weights,
                biases=softmax_biases,
                labels=label_ids,
                inputs=source_embeddings,
                num_sampled=sampled_count,
                num_classes=vocab_size,
                partition_strategy='div'
            ))
        else:  # LossFunction.SAMPLED_SOFTMAX == loss_func
            loss_op = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=softmax_weights,
                biases=softmax_biases,
                labels=label_ids,
                inputs=source_embeddings,
                num_sampled=sampled_count,
                num_classes=vocab_size,
                partition_strategy='div',
            ))

        assert mode == tf.estimator.ModeKeys.TRAIN

        opt_inst = get_optimizer_instance(train_opt, learning_rate=learn_rate)
        opt_inst = clip_gradients_by_norm(opt_inst, _CLIP_NORM)
        train_op = opt_inst.minimize(loss_op, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss_op, train_op=train_op)
