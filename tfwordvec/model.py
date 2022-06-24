from keras import layers, models
from nlpvocab import Vocabulary
from tfmiss.keras.layers import AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax, L2Scale, Reduction
from tfmiss.keras.layers import WordEmbedding, NgramEmbedding, BpeEmbedding, CnnEmbedding
from .config import RESERVED, UNK_MARK, InputUnit, VectModel, ModelHead


def build_model(config, unit_vocab, label_vocab, with_prep=False):
    inputs = _model_inputs(config, with_prep)
    logits = _model_encoder(config, unit_vocab, with_prep)(inputs)

    if config.l2_scale > 1.:
        logits = L2Scale(config.l2_scale, name='embedding_scale')(logits)

    top_labels, _ = label_vocab.split_by_frequency(config.label_freq)
    num_labels = len(top_labels)

    if ModelHead.SOFTMAX != config.model_head:
        inputs['labels'] = layers.Input(name='labels', shape=(), dtype='int32')

    if ModelHead.SAMPLED == config.model_head:
        head = SampledSofmax(num_labels, config.neg_samples)
        probs = head([logits, inputs['labels']])
    elif ModelHead.NCE == config.model_head:
        head = NoiseContrastiveEstimation(num_labels, config.neg_samples)
        probs = head([logits, inputs['labels']])
    elif ModelHead.ADAPTIVE == config.model_head:
        head = AdaptiveSoftmax(
            num_labels, list(config.asmax_cutoff), factor=config.asmax_factor, dropout=config.asmax_dropout)
        probs = head([logits, inputs['labels']])
    else:  # ModelHead.SOFTMAX == config.model_head:
        probs = layers.Dense(num_labels, name='logits')(logits)
        probs = layers.Activation('softmax', dtype='float32')(probs)

    model = models.Model(inputs=list(inputs.values()), outputs=probs, name='trainer')

    return model


def _model_encoder(config, unit_vocab, with_prep):
    inputs = _model_inputs(config, with_prep)
    embeddings = _unit_encoder(config, unit_vocab, with_prep)(inputs['units'])

    if VectModel.CBOWPOS == config.vect_model:
        positions = layers.Embedding(
            input_dim=config.window_size * 2,
            output_dim=config.embed_size,
            name='position_embedding')(inputs['positions'])
        embeddings = layers.multiply([embeddings, positions], name='position_encoding')

    if config.vect_model in {VectModel.CBOW, VectModel.CBOWPOS}:
        embeddings = Reduction('mean', name='context_reduction')(embeddings)

    return models.Model(inputs=list(inputs.values()), outputs=embeddings, name='context_encoder')


def _model_inputs(config, with_prep):
    has_context = int(config.vect_model in {VectModel.CBOW, VectModel.CBOWPOS})
    has_subwords = int(not with_prep and config.input_unit in {InputUnit.NGRAM, InputUnit.BPE, InputUnit.CNN})

    inputs = {
        'units': layers.Input(
            name='units',
            shape=(None,) * (int(has_context) + int(has_subwords)),
            dtype='string' if with_prep else 'int64',
            ragged=has_context or has_subwords)
    }

    if VectModel.CBOWPOS == config.vect_model:
        inputs['positions'] = layers.Input(name='positions', shape=(None,), dtype='int32', ragged=True)

    return inputs


def _unit_encoder(config, unit_vocab, with_prep):
    inputs = _model_inputs(config, with_prep)['units']
    embeddings = _unit_embedder(config, unit_vocab, with_prep)(inputs)

    return models.Model(inputs=inputs, outputs=embeddings, name='unit_encoder')


def _unit_embedder(config, vocabulary, with_prep):
    if isinstance(vocabulary, Vocabulary) and InputUnit.BPE == config.input_unit:
        unit_keys = vocabulary.tokens()
    elif isinstance(vocabulary, Vocabulary):
        unit_top, unit_unk = vocabulary.split_by_frequency(config.unit_freq)
        unit_top[UNK_MARK] += sum([f for _, f in unit_unk.most_common()])
        unit_keys = unit_top.tokens()
    else:
        assert isinstance(vocabulary, list), 'Vocabulary should be a "Vocabulary" instance or a list of tokens'
        unit_keys = vocabulary

    max_len = config.max_len if InputUnit.CHAR != config.input_unit else None
    common_kwargs = {
        'vocabulary': unit_keys,
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
        'with_prep': with_prep
    }

    if InputUnit.NGRAM == config.input_unit:
        return NgramEmbedding(
            minn=config.ngram_minn, maxn=config.ngram_maxn, itself=config.ngram_self.value,
            reduction=config.subword_comb.value, **common_kwargs)

    if InputUnit.BPE == config.input_unit:
        return BpeEmbedding(
            reduction=config.subword_comb.value, vocab_size=config.bpe_size, max_chars=config.bpe_chars,
            **common_kwargs)

    if InputUnit.CNN == config.input_unit:
        return CnnEmbedding(filters=list(config.cnn_filt), kernels=list(config.cnn_kern), **common_kwargs)

    return WordEmbedding(**common_kwargs)
