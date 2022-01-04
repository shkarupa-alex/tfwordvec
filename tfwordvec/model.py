import tensorflow as tf
from keras import layers, models
from tfmiss.keras.layers import AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax, L2Scale, Reduction
from tfmiss.keras.layers import WordEmbedding, CharNgramEmbedding, CharBpeEmbedding, CharCnnEmbedding, MapFlat
from .config import InputUnit, VectModel, ModelHead
from .input import RESERVED, UNK_MARK


def build_model(config, unit_vocab, label_vocab):
    inputs = _context_inputs(config)
    encoder = _build_encoder(config, unit_vocab)
    logits = encoder(inputs)

    top_labels, _ = label_vocab.split_by_frequency(config.label_freq)
    num_labels = len(top_labels)

    if ModelHead.SOFTMAX != config.model_head:
        inputs['labels'] = layers.Input(name='labels', shape=(), dtype=tf.int32)

    if ModelHead.SAMPLED == config.model_head:
        head = SampledSofmax(num_labels, config.neg_samples)
        probs = head([logits, inputs['labels']])
    elif ModelHead.NCE == config.model_head:
        head = NoiseContrastiveEstimation(num_labels, config.neg_samples)
        probs = head([logits, inputs['labels']])
    elif ModelHead.ADAPTIVE == config.model_head:
        head = AdaptiveSoftmax(num_labels, list(config.asmax_cutoff), config.asmax_factor, config.asmax_dropout)
        probs = head([logits, inputs['labels']])
    else:  # ModelHead.SOFTMAX == config.model_head:
        probs = layers.Dense(num_labels, name='logits')(logits)
        probs = layers.Activation('softmax', dtype=tf.float32)(probs)

    model = models.Model(inputs=list(inputs.values()), outputs=probs, name='trainer')

    return model


def _build_encoder(config, unit_vocab):
    inputs = _context_inputs(config)

    encoder = _unit_encoder(config, unit_vocab)
    embeddings = MapFlat(encoder, name='unit_encoding')(inputs['units'])

    if VectModel.CBOWPOS == config.vect_model:
        positions = layers.Embedding(
            input_dim=config.window_size * 2,
            output_dim=config.embed_size,
            name='position_embedding')(inputs['positions'])
        embeddings = layers.multiply([embeddings, positions], name='position_encoding')

    if config.vect_model in {VectModel.CBOW, VectModel.CBOWPOS}:
        embeddings = Reduction('mean', name='context_reduction')(embeddings)

    return models.Model(inputs=list(inputs.values()), outputs=embeddings, name='context_encoder')


def _context_inputs(config):
    has_context = int(config.vect_model in {VectModel.CBOW, VectModel.CBOWPOS})
    inputs = {
        'units': layers.Input(
            name='units',
            shape=(None,) * int(has_context),
            dtype=tf.string,
            ragged=has_context)
    }

    if VectModel.CBOWPOS == config.vect_model:
        inputs['positions'] = layers.Input(name='positions', shape=(None,), dtype=tf.int32, ragged=True)

    return inputs


def _unit_encoder(config, unit_vocab):
    inputs = layers.Input(name='units', shape=(), dtype=tf.string)

    unit_top, unit_unk = unit_vocab.split_by_frequency(config.unit_freq)
    unit_top[UNK_MARK] += sum([f for _, f in unit_unk.most_common()])
    unit_keys = unit_top.tokens()

    unit_embedder = _unit_embedder(config, unit_keys)
    embeddings = unit_embedder(inputs)

    if config.l2_scale > 1.:
        embeddings = L2Scale(config.l2_scale, name='embedding_scale')(embeddings)

    return models.Model(inputs=inputs, outputs=embeddings, name='unit_encoder')


def _unit_embedder(config, vocabulary):
    max_len = config.max_len if InputUnit.CHAR != config.input_unit else None
    common_kwargs = {
        'vocabulary': vocabulary,
        'output_dim': config.embed_size,
        'normalize_unicode': 'NFKC',
        'lower_case': config.lower_case,
        'zero_digits': config.zero_digits,
        'max_len': max_len,
        'reserved_words': RESERVED,
        'embed_type': config.embed_type.value,
        'adapt_cutoff': list(config.aembd_cutoff),
        'adapt_factor': config.aembd_factor,
        'name': 'unit_embedding',
        'show_warning': len(vocabulary) > len(RESERVED)
    }

    if InputUnit.NGRAM == config.input_unit:
        return CharNgramEmbedding(
            minn=config.ngram_minn, maxn=config.ngram_maxn, itself=config.ngram_self.value,
            reduction=config.ngram_comb.value, **common_kwargs)

    if InputUnit.BPE == config.input_unit:
        return CharBpeEmbedding(
            reduction=config.ngram_comb.value, vocab_size=config.bpe_size, max_chars=config.bpe_chars,
            **common_kwargs)

    if InputUnit.CNN == config.input_unit:
        return CharCnnEmbedding(filters=list(config.cnn_filt), kernels=list(config.cnn_kern), **common_kwargs)

    return WordEmbedding(**common_kwargs)
