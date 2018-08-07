import tensorflow as tf
import tensorflow.contrib.slim as slim
#import tflearn

def adversarial_discriminator(net, layers = [500, 500], scope='adversary', leaky=False, weight_decay = 2.5e-5, alpha = 0.1):
    if leaky:
        activation_fn = lambda x: tf.nn.leaky_relu(x, alpha = alpha)
    else:
        activation_fn = tf.nn.relu

    with tf.variable_scope(scope):
        with slim.arg_scope([slim.fully_connected],
                        activation_fn=activation_fn,
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
            for i, dim in enumerate(layers):
                net = slim.fully_connected(net, dim, scope = "fc_{}".format(i + 1))
            net = slim.fully_connected(net, 2, activation_fn=None)
    return net
