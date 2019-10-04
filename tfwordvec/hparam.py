from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfmiss.training import HParams


def build_hparams(custom=None):
    assert custom is None or isinstance(custom, dict)

    params = HParams(
        vector_model='cbow',  # or 'skipgram'
        model_unit='word',  # or 'char' or 'ngram'
        label_case='lower',  # or 'same'
        feat_case='lower',  # or 'same' or 'extract'
        zero_digits=False,
        word_mean=5.3,
        word_std=6.3,
        ngram_minn=3,
        ngram_maxn=5,
        ngram_itself='always',  # or 'asis', 'never', 'alone'
        min_freq=5,

        window_size=5,
        embed_size=256,
        sample_thold=1e-3,
        neg_samples=5,

        batch_size=256,
        bucket_cbow=True,
        num_repeats=1,
        train_loss='nce',  # or 'ss'
        train_optim='adam',
        learn_rate=0.05,
    )

    if custom is not None:
        params.override_from_dict(custom)

    params.vector_model = params.vector_model.lower()
    params.model_unit = params.model_unit.lower()
    params.label_case = params.label_case.lower()
    params.feat_case = params.feat_case.lower()
    params.train_loss = params.train_loss.lower()
    params.train_optim = params.train_optim.lower()

    assert params.vector_model in {'cbow', 'skipgram'}
    assert params.model_unit in {'char', 'word', 'ngram'}
    assert params.label_case in {'lower', 'same'}
    assert params.feat_case in {'lower', 'same', 'extract'}
    assert params.word_mean >= 0.
    assert params.word_std >= 0.
    assert params.ngram_maxn > params.ngram_minn > 0
    assert params.ngram_itself in {'always', 'asis', 'never', 'alone'}
    assert params.min_freq >= 0

    assert params.window_size > 0
    assert params.embed_size > 0
    assert params.sample_thold > 0.
    assert params.neg_samples > 0

    assert params.batch_size > 0
    assert params.num_repeats > 0
    assert params.train_loss in {'nce', 'ss'}
    assert len(params.train_optim) > 0
    assert params.learn_rate > 0.

    return params
