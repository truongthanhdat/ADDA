import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.model import register_model_fn
from nets.resnet_v1 import resnet_v1

@register_model_fn("resnet_v1_101")
def resnet_v1_101(images, scope = "resnet_v1_101", is_training = True, num_classes = 1000,
                    image_size = 224, weight_decay = 1e-5, reuse = False):

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay = weight_decay)):
        net, layers = resnet_v1.resnet_v1_101(images, num_classes = num_classes,
                            is_training = is_training, scope = scope, reuse = reuse)
    return net, layers

@register_model_fn("resnet_v1_50")
def resnet_v1_50(images, scope = "resnet_v1_50", is_training = True, num_classes = 1000,
                    image_size = 224, weight_decay = 1e-5, reuse = False):

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay = weight_decay)):
        net, layers = resnet_v1.resnet_v1_50(images, num_classes = num_classes,
                            is_training = is_training, scope = scope, reuse = reuse)
    return net, layers

@register_model_fn("resnet_v1_152")
def resnet_v1_152(images, scope = "resnet_v1_152", is_training = True, num_classes = 1000,
                    image_size = 224, weight_decay = 1e-5, reuse = False):

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay = weight_decay)):
        net, layers = resnet_v1.resnet_v1_152(images, num_classes = num_classes,
                            is_training = is_training, scope = scope, reuse = reuse)
    return net, layers
