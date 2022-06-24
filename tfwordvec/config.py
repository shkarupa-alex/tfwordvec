import os
from dataclasses import dataclass
from enum import Enum
from keras import optimizers as core_opt
from omegaconf import OmegaConf
from typing import Optional, List
from tensorflow_addons import optimizers as add_opt  # Required to initialize custom optimizers

UNK_MARK = '[UNK]'
BOS_MARK = '[BOS]'
EOS_MARK = '[EOS]'
RESERVED = [UNK_MARK, BOS_MARK, EOS_MARK]


class InputUnit(Enum):
    CHAR = 'char'
    WORD = 'word'
    NGRAM = 'ngram'
    BPE = 'bpe'
    CNN = 'cnn'


class NgramSelf(Enum):
    ALWAYS = 'always'
    ALONE = 'alone'


class SubwordComb(Enum):
    MIN = 'min'
    MAX = 'max'
    MEAN = 'mean'
    SUM = 'sum'
    PROD = 'prod'
    SQRTN = 'sqrtn'


class EmbedType(Enum):
    DENSE_AUTO = 'dense_auto'
    DENSE_CPU = 'dense_cpu'
    ADAPTIVE = 'adapt'


class ModelHead(Enum):
    SOFTMAX = 'softmax'
    SAMPLED = 'sampled_softmax'
    NCE = 'noise_contrastive'
    ADAPTIVE = 'adaptive_softmax'


class VectModel(Enum):
    SKIPGRAM = 'skipgram'
    CBOW = 'cbow'
    CBOWPOS = 'cbowpos'


@dataclass
class Config:
    # Input
    input_unit: InputUnit = InputUnit.WORD
    max_len: int = 32
    lower_case: bool = True
    zero_digits: bool = False
    unit_freq: int = 2

    ngram_minn: Optional[int] = 3
    ngram_maxn: Optional[int] = 5
    ngram_self: Optional[NgramSelf] = NgramSelf.ALWAYS

    bpe_size: Optional[int] = 32000
    bpe_chars: Optional[int] = 1000

    subword_comb: Optional[SubwordComb] = SubwordComb.MEAN

    cnn_filt: Optional[List[int]] = (32, 32, 64, 128, 256, 512, 1024)
    cnn_kern: Optional[List[int]] = (1, 2, 3, 4, 5, 6, 7)

    # Body
    embed_type: EmbedType = EmbedType.DENSE_CPU
    embed_size: int = 256
    aembd_cutoff: Optional[List[int]] = (2000,)
    aembd_factor: Optional[int] = 4
    l2_scale: float = 0.

    # Head
    model_head: ModelHead = ModelHead.SOFTMAX
    label_freq: int = 2
    neg_samples: Optional[int] = 5
    asmax_cutoff: Optional[List[int]] = (2000,)
    asmax_factor: Optional[int] = 4
    asmax_dropout: Optional[float] = 0.

    # Objective
    vect_model: VectModel = VectModel.SKIPGRAM
    window_size: int = 5
    samp_thold: float = 1e-3

    # Train
    batch_size: int = 256
    bucket_cbow: bool = True
    num_epochs: int = 5
    mixed_fp16: bool = False
    use_jit: bool = False
    train_optim: str = 'Adam'
    learn_rate: float = 0.05


def build_config(custom):
    default = OmegaConf.structured(Config)

    if isinstance(custom, str) and custom.endswith('.yaml') and os.path.isfile(custom):
        custom = OmegaConf.load(custom)
    else:
        assert isinstance(custom, dict), 'Bad custom config format'
        custom = OmegaConf.create(custom)

    merged = OmegaConf.merge(default, custom)
    OmegaConf.set_readonly(merged, True)

    # noinspection PyTypeChecker
    conf: Config = merged

    # Input
    assert conf.max_len is None or 3 < conf.max_len, 'Bad maximum word length'
    assert 0 < conf.unit_freq, 'Bad minimum unit frequency'
    if InputUnit.NGRAM == conf.input_unit:
        assert 0 < conf.ngram_minn <= conf.ngram_maxn, 'Bad min/max ngram sizes'
    if InputUnit.BPE == conf.input_unit:
        assert 0 < conf.bpe_size, 'Bad BPE vocabulary size'
        assert 0 < conf.bpe_chars, 'Bad BPE chars count'
    if InputUnit.CNN == conf.input_unit:
        assert conf.cnn_kern, 'Bad (empty) CNN kernels list'
        assert len(conf.cnn_kern) == len(conf.cnn_filt), 'Bad CNN filters length'

    # Body
    assert 0 < conf.embed_size, 'Bad embedding size'
    if EmbedType.ADAPTIVE == conf.embed_type:
        assert conf.aembd_cutoff, 'Bad adaptive embedding cutoff'
        assert 0 < conf.aembd_factor, 'Bad adaptive embedding factor'
    assert 0. <= conf.l2_scale, 'Bad l2 scale factor'

    # Head
    assert 0 < conf.label_freq, 'Bad minimum label frequency'
    if conf.model_head in {ModelHead.SAMPLED, ModelHead.NCE}:
        assert 0 < conf.neg_samples, 'Bad number of negative samples'
    if ModelHead.ADAPTIVE == conf.model_head:
        assert conf.asmax_cutoff, 'Bad adaptive softmax cutoff'
        assert 0 < conf.asmax_factor, 'Bad adaptive softmax factor'
        assert 0. <= conf.asmax_dropout < 1., 'Bad adaptive softmax dropout'

    # Objective
    assert 0 < conf.window_size, 'Bad window size'
    assert 0. < conf.samp_thold, 'Bad downsampling threshold'

    # Train
    assert 0 < conf.batch_size, 'Bad batch size'
    assert 0 < conf.num_epochs, 'Bad number of epochs'
    assert conf.train_optim, 'Bad train optimizer'
    assert 'ranger' == conf.train_optim.lower() or core_opt.get(conf.train_optim), \
        'Unsupported train optimizer'
    assert 0. < conf.learn_rate, 'Bad learning rate'

    # TODO: dense_cpu if ss or nce

    return conf
