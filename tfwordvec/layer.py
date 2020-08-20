from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Wrapper
from tensorflow.python.keras.layers.preprocessing.reduction import Reduction as _Reduction
from tensorflow.python.keras.utils import tf_utils


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
class MapFlat(Wrapper):
    def __init__(self, layer, **kwargs):
        super(MapFlat, self).__init__(layer, **kwargs)
        self.supports_masking = layer.supports_masking
        self._supports_ragged_inputs = True

    def call(self, inputs, **kwargs):
        return tf.ragged.map_flat_values(self.layer, inputs)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + self.layer.compute_output_shape([None])[1:]
