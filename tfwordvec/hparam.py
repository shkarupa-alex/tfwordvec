from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfmiss.training import HParams
from tensorflow.keras import optimizers as core_opt
from tensorflow_addons import optimizers as add_opt  # Required to initialize custom optimizers


def build_hparams(custom):
    if not isinstance(custom, dict):
        raise TypeError('Expected custom parameters to be dict. Got {}'.format(type(custom)))

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

    if params.input_unit not in {'char', 'word', 'ngram'}:
        raise ValueError('Unsupported input unit')
    if 0 >= params.unit_freq:
        raise ValueError('Bad minimum unit frequency')
    if 0 >= params.label_freq:
        raise ValueError('Bad minimum label frequency')
    if not 0 < params.ngram_minn < params.ngram_maxn:
        raise ValueError('Bad min/max ngram sizes')
    if params.ngram_self not in {'always', 'alone'}:
        raise ValueError('Unsupported ngram extractor')
    if params.ngram_comb not in {'mean', 'sum', 'min', 'max', 'prod'}:
        raise ValueError('Unsupported ngram combiner')

    if params.vect_model not in {'cbow', 'skipgram', 'cbowpos'}:
        raise ValueError('Unsupported vector model')
    if 0 >= params.window_size:
        raise ValueError('Bad window size')
    if 0. >= params.samp_thold:
        raise ValueError('Bad downsampling threshold')
    if 0 >= params.embed_size:
        raise ValueError('Bad embedding size')
    if params.embed_type not in {'dense', 'adapt'}:
        raise ValueError('Unsupported embedding type')
    if 'adapt' == params.embed_type and not params.aemb_cutoff:
        raise ValueError('Bad adaptive embedding cutoff')
    if 'adapt' == params.embed_type and 0 >= params.aemb_factor:
        raise ValueError('Bad adaptive embedding factor')
    if 0. > params.l2_scale:
        raise ValueError('Bad l2 scale factor')

    if params.model_head not in {'ss', 'nce', 'asm', 'sm'}:
        raise ValueError('Unsupported softmax head')
    if params.model_head in {'nce', 'ss'} and 0 >= params.neg_samp:
        raise ValueError('Bad number of negative samples')
    if 'asm' == params.model_head and not params.asm_cutoff:
        raise ValueError('Bad adaptive softmax cutoff')
    if 'asm' == params.model_head and 0 >= params.asm_factor:
        raise ValueError('Bad adaptive softmax factor')
    if 'asm' == params.model_head and not 0. <= params.asm_drop < 1.:
        raise ValueError('Bad adaptive softmax dropout')

    if 0 >= params.batch_size:
        raise ValueError('Bad batch size')
    if 0 >= params.num_epochs:
        raise ValueError('Bad number of epochs')
    if not len(params.train_optim):
        raise ValueError('Bad train optimizer')
    elif 'ranger' != params.train_optim.lower():
        try:
            core_opt.get(params.train_optim)
        except:
            raise ValueError('Unsupported train optimizer')
    if 0. >= params.learn_rate:
        raise ValueError('Bad learning rate')

    return params
