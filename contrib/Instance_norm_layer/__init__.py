import hls4ml
from pathlib import Path
from Instance_norm_layer.instance_norm import parse_instancenorm_layer
from Instance_norm_layer.instance_norm import InstanceNormalization
from Instance_norm_layer.instance_norm import InstanceNormConfigTemplate
from Instance_norm_layer.instance_norm import InstanceNormFunctionTemplate
from hls4ml.converters import get_supported_keras_layers


if 'Addons>InstanceNormalization' not in get_supported_keras_layers():
	# Register the converter for custom InstanceNorm
	hls4ml.converters.register_keras_layer_handler('Addons>InstanceNormalization', parse_instancenorm_layer)
	# Register the hls4ml's IR layer
	hls4ml.model.layers.register_layer('Addons>InstanceNormalization', InstanceNormalization)
	# Register the optimization passes (if any)
	backend = hls4ml.backends.get_backend('Vivado')
	# Register template passes for the given backend
	backend.register_template(InstanceNormConfigTemplate)
	backend.register_template(InstanceNormFunctionTemplate)
	# Register HLS implementation
	p = Path(__file__).parent / 'nnet_instancenorm.h'
	backend.register_source(p)
	print("Instance Normalization layer registered in hls4ml")

