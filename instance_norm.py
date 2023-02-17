"""
    Layer Instance Normalization
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from keras.layers.merge import _Merge as Merge
except Exception:
    from keras.layers.merging.base_merge import _Merge as Merge

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops

import hls4ml
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.model.attributes import ConfigurableAttribute, TypeAttribute
from hls4ml.model.types import FixedPrecisionType, RoundingMode, SaturationMode

# hls4ml implementation
class InstanceNormalization(hls4ml.model.layers.Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_filt', default=0),
        WeightAttribute('scale'),
        WeightAttribute('bias'),
        TypeAttribute('scale'),
        TypeAttribute('bias'),
    ]
    def initialize(self):
        inp = self.get_input_variable()
        
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

        gamma = self.model.get_weights_data(self.name, 'gamma')
        beta = self.model.get_weights_data(self.name, 'beta')
        
        mean = np.mean(gamma, axis=(1,2), keepdims=True)
        var = np.var(gamma, axis=(1,2), keepdims=True)

        scale = gamma / np.sqrt(var + self.get_attr('epsilon'))
        bias = beta - gamma * mean / np.sqrt(var + self.get_attr('epsilon'))

        self.add_weights_variable(name='scale', var_name='s{index}', data=scale)
        self.add_weights_variable(name='bias', var_name='b{index}', data=bias)

# Templates
# ???
instancenorm_config = """struct config{index} : nnet::instancenorm_config {{
    
    typedef {bias_t.name} bias_t;
    typedef {scale_t.name} scale_t;
    
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned n_scale_bias  = {n_scale_bias};

    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse_factor};
    static const bool store_weights_in_bram = {store_weights_in_bram};
    static const unsigned n_zeros = {n_zeros};
    
    static const unsigned table_size = {table_size};
    static constexpr float exp_range = {exp_range};
}};\n"""
instancenorm_function_template  = (
    'nnet::instancenorm<{data_t}, {res_t}, {CONFIG_T}>({data}, {res}, {scale}, {bias});'
)
instancenorm_include_list = ['nnet_utils/nnet_instancenorm.h']


class InstanceNormalizationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(InstanceNormalization)
        self.template = instancenorm_config

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_input_variable().size_cpp()

        return self.template.format(**params)


class InstanceNormalizationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(InstanceNormalization, include_header=bn_include_list)
        self.template = instancenorm_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)

# Parser for converter
# @keras_handler('InstanceNormalization')
def parse_instancenorm_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'InstanceNormalization' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
    layer['n_in'] = in_size
    layer['n_out'] = layer['n_in']
    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) == 3:
        layer['n_filt'] = input_shapes[0][2]
    elif len(input_shapes[0]) == 4:
        layer['n_filt'] = input_shapes[0][3]

    return layer, [shape for shape in input_shapes[0]]


def main():
    
