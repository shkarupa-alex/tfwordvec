from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import InputSpec, Layer, Wrapper
from tensorflow.python.keras.layers.preprocessing.reduction import Reduction as _Reduction
from tensorflow.python.keras.utils import tf_utils
from tfmiss.text import wrap_with, char_ngrams, normalize_unicode, lower_case, zero_digits
from tfmiss.preprocessing import cbow_infer
from .input import BOS_MARK, EOS_MARK, UNK_MARK, RESERVED


@tf.keras.utils.register_keras_serializable(package='WordVec')
class ExpandNgams(Layer):
    def __init__(self, ngram_minn, ngram_maxn, ngram_self, reserved=RESERVED, *args, **kwargs):
        super(ExpandNgams, self).__init__(*args, **kwargs)
        self.input_spec = InputSpec(dtype='string')
        self._supports_ragged_inputs = True

        self.ngram_minn = ngram_minn
        self.ngram_maxn = ngram_maxn
        self.ngram_self = ngram_self
        self.reserved = reserved

    def call(self, inputs, **kwargs):
        outputs = wrap_with(inputs, '<', '>', skip=self.reserved)
        outputs = char_ngrams(outputs, self.ngram_minn, self.ngram_maxn, self.ngram_self, skip=self.reserved)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + (None,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'ngram_minn': self.ngram_minn,
            'ngram_maxn': self.ngram_maxn,
            'ngram_self': self.ngram_self,
            'reserved': self.reserved
        })

        return config


@tf.keras.utils.register_keras_serializable(package='WordVec')
class MapFlat(Wrapper):
    def __init__(self, layer, **kwargs):
        super(MapFlat, self).__init__(layer, **kwargs)
        self._supports_ragged_inputs = True

    def call(self, inputs, **kwargs):
        return tf.ragged.map_flat_values(self.layer, inputs)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + self.layer.compute_output_shape([None])[1:]


@tf.keras.utils.register_keras_serializable(package='WordVec')
class Reduction(_Reduction):
    def __init__(self, *args, **kwargs):
        super(Reduction, self).__init__(*args, **kwargs)
        self._supports_ragged_inputs = True

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:self.axis] + input_shape[self.axis + 1:]

    def get_config(self):
        config = super(Reduction, self).get_config()
        config.update({
            'reduction': self.reduction,
            'axis': self.axis,
        })

        return config


@tf.keras.utils.register_keras_serializable(package='WordVec')
class NormalizeUnits(Layer):
    def __init__(self, lower, zero, **kwargs):
        super(NormalizeUnits, self).__init__(**kwargs)
        self.input_spec = InputSpec(dtype='string')
        self._supports_ragged_inputs = True

        self.lower = lower
        self.zero = zero

    def call(self, inputs, **kwargs):
        outputs = normalize_unicode(inputs, 'NFKC')
        if self.lower:
            outputs = lower_case(outputs)
        if self.zero:
            outputs = zero_digits(outputs)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(NormalizeUnits, self).get_config()
        config.update({
            'lower': self.lower,
            'zero': self.zero,
        })

        return config


@tf.keras.utils.register_keras_serializable(package='WordVec')
class CbowContext(Wrapper):
    def __init__(self, layer, window, position, **kwargs):
        super(CbowContext, self).__init__(layer, **kwargs)
        self.input_spec = InputSpec(dtype='string', ndim=2)
        self._supports_ragged_inputs = True

        self.window = window
        self.position = position

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, tf.RaggedTensor):
            inputs = tf.RaggedTensor.from_tensor(inputs, ragged_rank=1)  # inputs.shape.rank - 1

        bos = tf.fill([inputs.nrows(), 1], BOS_MARK)
        eos = tf.fill([inputs.nrows(), 1], EOS_MARK)
        sources = tf.concat([bos, inputs, eos], axis=1)

        features = {}
        outputs, positions = cbow_infer(sources, self.window, UNK_MARK)
        if self.position:
            positions = tf.ragged.map_flat_values(
                lambda flat: tf.where(
                    tf.greater(flat, 0),
                    flat - 1,
                    flat
                ) + self.window,
                positions)
            features['positions'] = positions

        features['units'] = outputs
        outputs = self.layer.call(features)
        outputs = sources.with_flat_values(outputs)
        outputs = outputs[:, 1:-1, :]

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + self.layer.compute_output_shape(input_shape)[1:]

    def get_config(self):
        config = super(CbowContext, self).get_config()
        config.update({
            'window': self.window,
            'position': self.position,
        })

        return config
