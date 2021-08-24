import tensorflow as tf
from keras import layers, models
from tfmiss.keras.layers import AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax, L2Scale, Reduction
from tfmiss.keras.layers import WordEmbedding, CharNgramEmbedding, CharBpeEmbedding, CharCnnEmbedding, MapFlat
from .input import RESERVED, UNK_MARK


def build_model(h_params, unit_vocab, label_vocab):
    inputs = _context_inputs(h_params)
    encoder = _build_encoder(h_params, unit_vocab)
    logits = encoder(inputs)

    top_labels, _ = label_vocab.split_by_frequency(h_params.label_freq)
    num_labels = len(top_labels)

    if 'sm' != h_params.model_head:
        inputs['labels'] = layers.Input(name='labels', shape=(), dtype=tf.int32)

    if 'ss' == h_params.model_head:
        head = SampledSofmax(num_labels, h_params.neg_samp)
        probs = head([logits, inputs['labels']])
    elif 'nce' == h_params.model_head:
        head = NoiseContrastiveEstimation(num_labels, h_params.neg_samp)
        probs = head([logits, inputs['labels']])
    elif 'asm' == h_params.model_head:
        head = AdaptiveSoftmax(num_labels, h_params.asm_cutoff, h_params.asm_factor, h_params.asm_drop)
        probs = head([logits, inputs['labels']])
    else:  # 'sm' == h_params.model_head:
        probs = layers.Dense(num_labels, name='logits')(logits)
        probs = layers.Activation('softmax', dtype=tf.float32)(probs)

    model = models.Model(inputs=list(inputs.values()), outputs=probs, name='trainer')

    return model


def _build_encoder(h_params, unit_vocab):
    inputs = _context_inputs(h_params)

    encoder = _unit_encoder(h_params, unit_vocab)
    embeddings = MapFlat(encoder, name='unit_encoding')(inputs['units'])

    if 'cbowpos' == h_params.vect_model:
        positions = layers.Embedding(
            input_dim=h_params.window_size * 2,
            output_dim=h_params.embed_size,
            name='position_embedding')(inputs['positions'])
        embeddings = layers.multiply([embeddings, positions], name='position_encoding')

    if h_params.vect_model in {'cbow', 'cbowpos'}:
        embeddings = Reduction('mean', name='context_reduction')(embeddings)

    return models.Model(inputs=list(inputs.values()), outputs=embeddings, name='context_encoder')


def _context_inputs(h_params):
    has_context = int(h_params.vect_model in {'cbow', 'cbowpos'})
    inputs = {
        'units': layers.Input(
            name='units',
            shape=(None,) * int(has_context),
            dtype=tf.string,
            ragged=has_context)
    }

    if 'cbowpos' == h_params.vect_model:
        inputs['positions'] = layers.Input(name='positions', shape=(None,), dtype=tf.int32, ragged=True)

    return inputs


def _unit_encoder(h_params, unit_vocab):
    inputs = layers.Input(name='units', shape=(), dtype=tf.string)

    unit_top, unit_unk = unit_vocab.split_by_frequency(h_params.unit_freq)
    unit_top[UNK_MARK] += sum([f for _, f in unit_unk.most_common()])
    unit_keys = unit_top.tokens()

    unit_embedder = _unit_embedder(h_params, unit_keys)
    embeddings = unit_embedder(inputs)

    if h_params.l2_scale > 1.:
        embeddings = L2Scale(h_params.l2_scale, name='embedding_scale')(embeddings)

    return models.Model(inputs=inputs, outputs=embeddings, name='unit_encoder')


def _unit_embedder(h_params, vocabulary):
    common_kwargs = {
        'vocabulary': vocabulary,
        'output_dim': h_params.embed_size,
        'normalize_unicode': 'NFKC',
        'lower_case': h_params.lower_case,
        'zero_digits': h_params.zero_digits,
        'max_len': h_params.max_len,
        'reserved_words': RESERVED,
        'embed_type': h_params.embed_type,
        'adapt_cutoff': h_params.aemb_cutoff,
        'adapt_factor': h_params.aemb_factor,
        'name': 'unit_embedding',
        'show_warning': len(vocabulary) > len(RESERVED)
    }

    if 'ngram' == h_params.input_unit:
        return CharNgramEmbedding(
            minn=h_params.ngram_minn, maxn=h_params.ngram_maxn, itself=h_params.ngram_self,
            reduction=h_params.ngram_comb, **common_kwargs)

    if 'bpe' == h_params.input_unit:
        return CharBpeEmbedding(
            reduction=h_params.ngram_comb, vocab_size=h_params.bpe_size, max_chars=h_params.bpe_chars,
            **common_kwargs)

    if 'cnn' == h_params.input_unit:
        return CharCnnEmbedding(filters=h_params.cnn_filt, kernels=h_params.cnn_kern, **common_kwargs)

    return WordEmbedding(**common_kwargs)
