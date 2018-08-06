import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
from preprocessing.vgg_preprocessing import preprocess_image
from datasets import dataset_factory
from collections import OrderedDict
import os

class Timer:
    def __init__(self):
        pass

    def tic(self):
        self._tic = time.time()

    def toc(self):
        return time.time() - self._tic

def get_batch_from_dataset(dataset = None, batch_size = 32, num_readers = 32,
                            image_size = 224, num_preprocessing_threads = 4,
                            is_training = True, verbose = True, gray2rgb = 0):
    assert dataset is not None, "Please Provide Dataset"

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                num_readers = num_readers,
                                                common_queue_capacity = 20 * batch_size,
                                                common_queue_min = 10 * batch_size)

    [image, label] = provider.get(["image", "label"])

    if gray2rgb == 1:
        image = tf.image.grayscale_to_rgb(image)
    elif gray2rgb == -1:
        image = tf.image.rgb_to_grayscale(image)

    image = preprocess_image(image,
                    output_height = image_size, output_width = image_size,
                    is_training = is_training)

    if len(image.get_shape().as_list()) != 3:
        image = tf.expand_dims(image, axis = -1)
    assert len(image.get_shape().as_list()) == 3, "Wrong Format of Image Input"

    image =  image / 255
    images, labels = tf.train.batch([image, label],
                                batch_size = batch_size,
                                num_threads = num_preprocessing_threads,
                                capacity = 5 * batch_size)

    if verbose:
        print("Dataset has {} images and {} classes".format(dataset.num_samples, dataset.num_classes))

    return images, labels

def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict

