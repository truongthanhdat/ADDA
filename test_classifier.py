import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import utils
from datasets.dataset_factory import get_dataset
import argparse
from nets.model import get_model_fn
def parse_args():
    parser = argparse.ArgumentParser("Testing Classifier")

    #Dataset Configuration
    parser.add_argument("--image_size", help = "Image Size", type = int, default = 224)
    parser.add_argument("--gray2rgb", help = "Convert Image between RGB and Grayscale:  0: keep; 1: gray2rgb; -1: rgb2gray", type = int, default = 0)
    parser.add_argument("--dataset", help = "Dataset Name", type = str, default = "usps")
    parser.add_argument("--split", help = "Split Name", type = str, default = "test")
    parser.add_argument("--dataset_dir", help = "Dataset Dir", type = str, default = "data/usps")

    #Learning Configuration
    parser.add_argument("--model", help = "Model Name", type = str, default = "lenet")
    parser.add_argument("--num_iters", help = "Number of Iterations", type = int, default = 10000)
    parser.add_argument("--batch_size", help = "Batch Size", type = int, default = 32)
    parser.add_argument("--model_path", help = "Model Path", type = str, default = "./model/pretrained")
    parser.add_argument("--num_readers", help = "Number of Readers", type = int, default = 4)
    parser.add_argument("--num_preprocessing_threads", help = "Number of Preprocessing Threads", type = int, default = 4)

    return parser.parse_args()

def main():
    #####################
    ## Parse Arguments ##
    #####################
    options = parse_args()

    print("Configurations")
    for var in vars(options):
        print("\t>> {}: {} ".format(var, getattr(options, var)))
    print("Enter to Continue or Ctrl+C to Break")

    ##################
    ## Load dataset ##
    ##################
    dataset = get_dataset(options.dataset, options.split, options.dataset_dir)
    images, labels = utils.get_batch_from_dataset(dataset = dataset, batch_size = options.batch_size,
                                    num_readers = options.num_readers, is_training = False,
                                    num_preprocessing_threads = options.num_preprocessing_threads,
                                    gray2rgb = options.gray2rgb, verbose = True, image_size = options.image_size)

    ################
    ## Load Model ##
    ################
    model_fn = get_model_fn(options.model)
    net, layers = model_fn(images, image_size = options.image_size, num_classes = dataset.num_classes,
                            is_training = False, scope = options.model)

    accuracy = slim.metrics.accuracy(tf.argmax(net, axis = 1), tf.cast(labels, tf.int64))


    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    model_path = tf.train.latest_checkpoint(options.model_path)
    if model_path is not None:
        saver.restore(sess, model_path)
        print("Restore model succcessfully from {}".format(model_path))
    else:
        raise ValueError("Cannot Found model from {}".format(options.model_path))

    #############
    ## Testing ##
    #############
    timer = utils.Timer()
    num_batchs = int(dataset.num_samples / options.batch_size) + 1
    total = 0.0
    for iter in xrange(1, num_batchs + 1):
        timer.tic()
        acc =  sess.run(accuracy)
        print("Iteration [{}/{}]:".format(iter, num_batchs))
        print("\t>> Accuracy:\t{}".format(acc))
        print("\t>> Executed Time:\t{} sec/iter".format(timer.toc()))

        total += acc

    print("Overal Accuracy: {}".format(total / num_batchs))

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == "__main__":
    main()



