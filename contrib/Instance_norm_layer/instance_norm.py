"""
    Usage example for a InstanceNormalization layer.
    It takes as imput: gamma and beta and computes scale and bias.

    The HLS part is in contrib/instance_norm/nnet_instancenorm.h
"""

import numpy as np

import sys

import hls4ml 
from hls4ml.model.attributes import Attribute
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate
# from hls4ml.backends import backend
from hls4ml import backends

from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.model.attributes import ConfigurableAttribute, TypeAttribute, WeightAttribute
from hls4ml.model.types import FixedPrecisionType, RoundingMode, SaturationMode


# hls4ml implementation
class InstanceNormalization(hls4ml.model.layers.Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_filt', default=0),
        Attribute('epsilon', value_type=float, default=1e-3),
        Attribute('ops_t', value_type=str, default='ap_fixed<32,10>'),
        WeightAttribute('gamma'),
        WeightAttribute('beta'),
        TypeAttribute('gamma'),
        TypeAttribute('beta')
    ]
    def initialize(self):
        inp = self.get_input_variable()
        
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

        gamma = self.model.get_weights_data(self.name, 'gamma')
        beta = self.model.get_weights_data(self.name, 'beta')

        self.add_weights_variable(name='gamma', var_name='gamma{index}', data=gamma)
        self.add_weights_variable(name='beta', var_name='beta{index}', data=beta)
       
# Templates
instancenorm_config = """struct config{index} : nnet::instancenorm_config {{
    
    typedef float eps_t;
    typedef {gamma_t.name} gamma_t;
    typedef {beta_t.name} beta_t;
    typedef {ops_t} ops_t;
    typedef ap_fixed<32,12> sum_t;

    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    eps_t eps = {epsilon};

    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse_factor};
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    
}};\n"""
# typedef {gamma_t.name} gamma_t;
#     typedef {beta_t.name} beta_t;

instancenorm_function_template  = (
    'nnet::instancenorm<{data_t}, {res_t}, {CONFIG_T}>({data}, {res}, {gamma}, {beta},{epsilon});'
)
instancenorm_include_list = ['nnet_utils/nnet_instancenorm.h']


class InstanceNormConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(InstanceNormalization)
        self.template = instancenorm_config

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_input_variable().size_cpp()
        # print(params)
        return self.template.format(**params)


class InstanceNormFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(InstanceNormalization, include_header=instancenorm_include_list)
        self.template = instancenorm_function_template

    def format(self, node):
        # {data_t}, {res_t}, {CONFIG_T}>({data}, {res}, {scale}, {bias}
        params = self._default_function_params(node)
        # params = {}
        # params.update(layer.attributes)
        # params['config'] = 'config{}'.format(layer.index)

        params['CONFIG_T']= f'config{node.index}'
        params['data_t'] = node.get_input_variable().type.name
        params['res_t'] = node.get_output_variable().type.name
        params['data'] = node.get_input_variable().name
        params['res'] = node.get_output_variable().name
        params['gamma'] = node.get_weights('gamma').name
        params['beta'] = node.get_weights('beta').name
        params['epsilon'] = node.get_attr('epsilon')

        # params = self._default_function_params(node)
        return self.template.format(**params)

# Parser for converter
def parse_instancenorm_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'InstanceNormalization' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)
    print(layer)

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


