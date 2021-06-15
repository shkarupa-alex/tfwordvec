import os
import json
from tfmiss.training import HParams
from tensorflow.keras import optimizers as core_opt
from tensorflow_addons import optimizers as add_opt  # Required to initialize custom optimizers


def build_hparams(custom):
    if isinstance(custom, str) and custom.endswith('.json') and os.path.isfile(custom):
        with open(custom, 'r') as file:
            custom = json.loads(file.read())

    assert isinstance(custom, dict), 'Bad hyperparameters format'

    params = HParams(
        input_unit='word',  # or 'char' or 'ngram'
        lower_case=True,
        zero_digits=False,
        unit_freq=1,
        label_freq=1,
        ngram_minn=3,
        ngram_maxn=5,
        ngram_self='always',  # or 'alone'
        ngram_comb='mean',  # or 'sum' or 'min' or 'max' or 'prod'

        vect_model='cbow',  # or 'skipgram' or 'cbowpos'
        window_size=5,
        samp_thold=1e-3,
        embed_size=256,
        embed_type='dense',  # or 'adapt'
        aemb_cutoff=[0],
        aemb_factor=4,
        l2_scale=0.,

        model_head='ss',  # or 'nce' or 'asm' or 'sm'
        neg_samp=5,
        asm_cutoff=[0],
        asm_factor=4,
        asm_drop=0.,

        batch_size=256,
        bucket_cbow=True,
        num_epochs=5,
        mixed_fp16=False,
        use_jit=False,
        train_optim='Adam',
        learn_rate=0.05,
    )
    params.embed_cutoff = []
    params.head_cutoff = []
    params.override_from_dict(custom)

    params.input_unit = params.input_unit.lower()
    params.ngram_self = params.ngram_self.lower()
    params.ngram_comb = params.ngram_comb.lower()
    params.vect_model = params.vect_model.lower()
    params.embed_type = params.embed_type.lower()
    params.model_head = params.model_head.lower()
    # Disabled to use tensorflow-addons optimizers
    # params.train_optim = params.train_optim.lower()

    assert params.input_unit in {'char', 'word', 'ngram'}, 'Unsupported input unit'
    assert 0 < params.unit_freq, 'Bad minimum unit frequency'
    assert 0 < params.label_freq, 'Bad minimum label frequency'
    assert 0 < params.ngram_minn < params.ngram_maxn, 'Bad min/max ngram sizes'
    assert params.ngram_self in {'always', 'alone'}, 'Unsupported ngram extractor'
    assert params.ngram_comb in {'mean', 'sum', 'min', 'max', 'prod'}, 'Unsupported ngram combiner'

    assert params.vect_model in {'cbow', 'skipgram', 'cbowpos'}, 'Unsupported vector model'
    assert 0 < params.window_size, 'Bad window size'
    assert 0. < params.samp_thold, 'Bad downsampling threshold'
    assert 0 < params.embed_size, 'Bad embedding size'
    assert params.embed_type in {'dense', 'adapt'}, 'Unsupported embedding type'
    if 'adapt' == params.embed_type:
        assert params.aemb_cutoff, 'Bad adaptive embedding cutoff'
        assert 0 < params.aemb_factor, 'Bad adaptive embedding factor'
    assert 0. <= params.l2_scale, 'Bad l2 scale factor'

    assert params.model_head in {'ss', 'nce', 'asm', 'sm'}, 'Unsupported softmax head'
    if params.model_head in {'nce', 'ss'}:
        assert 0 < params.neg_samp, 'Bad number of negative samples'
    if 'asm' == params.model_head:
        assert params.asm_cutoff, 'Bad adaptive softmax cutoff'
        assert 0 < params.asm_factor, 'Bad adaptive softmax factor'
        assert 0. <= params.asm_drop < 1., 'Bad adaptive softmax dropout'

    assert 0 < params.batch_size, 'Bad batch size'
    assert 0 < params.num_epochs, 'Bad number of epochs'
    assert params.train_optim, 'Bad train optimizer'
    assert 'ranger' == params.train_optim.lower() or core_opt.get(params.train_optim), 'Unsupported train optimizer'
    assert 0. < params.learn_rate, 'Bad learning rate'

    return params
