import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.model import register_model_fn
from collections import OrderedDict

@register_model_fn("lenet")
def lenet(images, scope = "lenet", is_training = True, num_classes = 10,
                image_size = 28, weight_decay = 2.5e-5, reuse=False):

    layers = OrderedDict()
    net = images
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected, slim.conv2d],
                            activation_fn = tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d], padding='VALID'):
                net = slim.conv2d(net, 20, 5, scope='conv1')
                layers['conv1'] = net
                net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
                layers['pool1'] = net
                net = slim.conv2d(net, 50, 5, scope='conv2')
                layers['conv2'] = net
                net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
                layers['pool2'] = net
                net = tf.contrib.layers.flatten(net)
                net = slim.fully_connected(net, 500, scope='fc3')
                layers['fc3'] = net
                net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc4')
                layers['fc4'] = net
    return net, layers

lenet.image_size  = 28
