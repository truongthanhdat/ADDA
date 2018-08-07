import tensorflow as tf
import tensorflow.contrib.slim as slim
from datasets.dataset_factory import get_dataset
import utils
import os
import argparse
from nets.model import get_model_fn
from nets.adversary import adversarial_discriminator
def parse_args():
    parser = argparse.ArgumentParser(description = "Training Adversarial Discriminator Domain Adaptation")
    #Dataset Configuration
    parser.add_argument("--source_dataset", help = "Source Dataset Name", type = str, default = "usps")
    parser.add_argument("--target_dataset", help = "Target Dataset Name", type = str, default = "mnist")

    parser.add_argument("--source_dataset_dir", help = "Source Dataset Directory", type = str, default = "data/usps")
    parser.add_argument("--target_dataset_dir", help = "Target Dataset Directory", type = str, default = "data/mnist")

    parser.add_argument("--source_gray2rgb", help = "Convert Between RGB and grayscale: 0: Keep; 1: gray2rgb; -1: rgb2gray", type = int, default = 0)
    parser.add_argument("--target_gray2rgb", help = "Convert Between RGB and grayscale: 0: Keep; 1: gray2rgb; -1: rgb2gray", type = int, default = 0)
    parser.add_argument("--num_readers", help = "Number of Readers", type = int, default = 4)
    parser.add_argument("--image_size", help = "Image Size", type = int, default = 224)
    parser.add_argument("--num_preprocessing_threads", help = "Number of Prerocessing Threads", default = 4)
    parser.add_argument("--split", help = "Split of Dataset", type = str, default = "train")

    #Model Path Configuration
    parser.add_argument("--source_model_path", help = "Source Model Path", type = str, default = "./model/pretrained")
    parser.add_argument("--target_model_path", help = "Target Model Path", type = str, default = "./model/target")
    parser.add_argument("--adversary_model_path", help = "Adeversarial Descriminator Model Path", type = str, default = "./model/adversary")

    #Learning Configuration
    parser.add_argument("--model", help = "Model", type = str, default = "lenet")
    parser.add_argument("--adversary_leaky", help = "Adversary Leaky", type = bool, default = False)
    parser.add_argument("--learning_rate_discriminator", help = "Learning Rate", type = float, default = 1e-5)
    parser.add_argument("--learning_rate_generator", help = "Learning Rate of Genertor", type = float, default = 1e-6)
    parser.add_argument("--batch_size", help = "Batch Size", type = int, default = 16)
    parser.add_argument("--num_iters", help = "Number of Iterations", type = int, default = 30000)
    parser.add_argument("--solver", help = "Choice optimizer", type = str, default = "adam")
    parser.add_argument("--checkpoint_steps", help = "Checkpoint Step", type = int, default = 100)
    parser.add_argument("--lr_decay_steps", help = "Decay Steps of Learning Rate", type = int, default = None)
    parser.add_argument("--lr_decay_rate", help = "Decay Rate", type = float, default = 0.1)
    parser.add_argument("--feature_name", help = "Feature Name", type = str, default = None)
    return parser.parse_args()

def main():
    #####################
    ## Parse Arguments ##
    #####################

    options = parse_args()
    print("Configurations")
    for var in vars(options):
        print("\t>> {}: {} ".format(var, getattr(options, var)))
    #print("Enter to Continue or Ctrl+C to Break")

    ##################
    ## Load dataset ##
    ##################
    source_dataset = get_dataset(options.source_dataset, options.split, options.source_dataset_dir)
    target_dataset = get_dataset(options.target_dataset, options.split, options.target_dataset_dir)

    source_images, source_labels = utils.get_batch_from_dataset(dataset = source_dataset, batch_size = options.batch_size,
                                    num_readers = options.num_readers, is_training = True,
                                    num_preprocessing_threads = options.num_preprocessing_threads,
                                    gray2rgb = options.source_gray2rgb, verbose = True, image_size = options.image_size)

    target_images, target_labels = utils.get_batch_from_dataset(dataset = target_dataset, batch_size = options.batch_size,
                                    num_readers = options.num_readers, is_training = True,
                                    num_preprocessing_threads = options.num_preprocessing_threads,
                                    gray2rgb = options.target_gray2rgb, verbose = True, image_size = options.image_size)

    ##################
    ## Model Config ##
    ##################
    model_fn = get_model_fn(options.model)

    source_ft, source_layers = model_fn(source_images, scope = "source_network", is_training = False,
                                        num_classes = source_dataset.num_classes, image_size = options.image_size)
    target_ft, target_layers = model_fn(target_images, scope = "target_network", is_training = True,
                                        num_classes = target_dataset.num_classes, image_size = options.image_size)
    source_pred = source_ft
    target_pred = target_ft

    if options.feature_name == "None" or options.feature_name is None:
        feature_name = None
    else:
        feature_name = options.feature_name

    if (feature_name is not None) and (feature_name in source_layers) and (feature_name in target_layers):
        source_ft = source_layers[options.feature_name]
        target_ft = target_layers[options.feature_name]
    elif feature_name is not None:
        raise ValueError("{} was not found in scope".format(feature_name))


    source_ft = tf.reshape(source_ft, [-1, int(source_ft.get_shape()[-1])])
    target_ft = tf.reshape(target_ft, [-1, int(target_ft.get_shape()[-1])])

    adversary_ft = tf.concat([source_ft, target_ft], 0)

    source_adversary_label = tf.zeros([tf.shape(source_ft)[0]], tf.int32)
    target_adversary_label = tf.ones([tf.shape(target_ft)[0]], tf.int32)
    adversary_label = tf.concat([source_adversary_label, target_adversary_label], 0)

    adversary_logits = adversarial_discriminator(adversary_ft, leaky = options.adversary_leaky, layers = [512, 500])
    mapping_loss = tf.losses.sparse_softmax_cross_entropy(1 - adversary_label, adversary_logits)
    adversary_loss = tf.losses.sparse_softmax_cross_entropy(adversary_label, adversary_logits)

    domain_accuracy = slim.metrics.accuracy(tf.argmax(adversary_logits, axis = 1), tf.cast(adversary_label, tf.int64))
    source_accuracy = slim.metrics.accuracy(tf.argmax(source_pred, axis = 1), tf.cast(source_labels, tf.int64))
    target_accuracy = slim.metrics.accuracy(tf.argmax(target_pred, axis = 1), tf.cast(target_labels, tf.int64))


    source_vars = utils.collect_vars("source_network", prepend_scope = options.model)
    target_vars = utils.collect_vars("target_network", prepend_scope = options.model)
    adversary_vars = utils.collect_vars("adversary", prepend_scope = "adversary")

    source_model_path = tf.train.latest_checkpoint(options.source_model_path)
    target_model_path = tf.train.latest_checkpoint(options.target_model_path)
    adversary_model_path = tf.train.latest_checkpoint(options.adversary_model_path)
    if source_model_path is None:
        raise ValueError("{} not found to restore source model".format(options.source_model_path))

    source_saver = tf.train.Saver(source_vars)
    target_saver = tf.train.Saver(target_vars)
    adversary_saver = tf.train.Saver(adversary_vars)

    learning_rate_generator_op  = tf.Variable(options.learning_rate_generator, name='learning_rate_generator', trainable=False)
    learning_rate_discriminator_op  = tf.Variable(options.learning_rate_discriminator, name='learning_rate_discriminator', trainable=False)

    if options.solver == 'sgd':
        generator_optimizer = tf.train.MomentumOptimizer(learning_rate_generator_op, 0.99, name = "generator_discriminator")
        discriminator_optimizer = tf.train.MomentumOptimizer(learning_rate_discriminator_op, 0.99, name = "discriminator_optimizer")
    elif options.solver == "adam":
        generator_optimizer = tf.train.AdamOptimizer(learning_rate_generator_op, 0.5, name = "generator_discriminator")
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate_discriminator_op, 0.5, name = "discriminator_optimizer")


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "target_network")
    if update_ops:
        with tf.control_dependencies(update_ops):
            mapping_step = generator_optimizer.minimize(mapping_loss, var_list=list(target_vars.values()))
    else:
        mapping_step = generator_optimizer.minimize(mapping_loss, var_list=list(target_vars.values()))
    adversary_step = discriminator_optimizer.minimize(adversary_loss, var_list=list(adversary_vars.values()))

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    source_saver.restore(sess, source_model_path)
    print("Restore Source Model from {}".format(source_model_path))
    if target_model_path is not None:
        target_saver.restore(sess, target_model_path)
        print("Restore Target Model from {}".format(target_model_path))
    else:
        target_saver.restore(sess, source_model_path)
        print("Restore Target Model from {}".format(source_model_path))
        target_model_path = os.path.join(options.target_model_path, "target_model_{}_to_{}_{}.ckpt".format(options.source_dataset, options.target_dataset, options.model))

    if adversary_model_path is not None:
        adversary_saver.restore(sess, adversary_model_path)
        print("Restore Adversarial Discrimintor from {}".format(adversary_model_path))
    else:
        print("Cannot Restore Adversarial Discriminator. It will be set to random.")
        adversary_model_path = os.path.join(options.adversary_model_path, "discriminator_model_{}_to_{}_{}.ckpt".format(options.source_dataset, options.target_dataset, options.model))


    sumaries = [
            tf.summary.image("Image: Source Image", source_images),
            tf.summary.image("Image: Target Image", target_images),
            tf.summary.scalar("Loss: Mapping Loss", mapping_loss),
            tf.summary.scalar("Loss: Adversary Loss", adversary_loss),
            tf.summary.scalar("Accuracy: Domain Accuracy", domain_accuracy),
            tf.summary.scalar("Accuracy: Source Accuracy", source_accuracy),
            tf.summary.scalar("Accuracy: Target Accuracy", target_accuracy)
            ]
    summary_op = tf.summary.merge(sumaries)
    summary_writer = tf.summary.FileWriter(options.adversary_model_path, graph=tf.get_default_graph())

    ##############
    ## Training ##
    ##############
    timer = utils.Timer()
    current_learning_rate_generator = options.learning_rate_generator
    current_learning_rate_discriminator = options.learning_rate_discriminator
    for iter in xrange(1, options.num_iters + 1):
        timer.tic()
        _mapping_loss, _adversary_loss, _, _, src_acc, tgt_acc, dom_acc, summary, lr_gen, lr_dis  = sess.run([
                                                mapping_loss, adversary_loss,
                                                mapping_step, adversary_step,
                                                source_accuracy, target_accuracy, domain_accuracy,
                                                summary_op, learning_rate_generator_op, learning_rate_discriminator_op])

        summary_writer.add_summary(summary, iter)
        print("Iteration [{}/{}]:".format(iter, options.num_iters))
        print("\t>> Mapping Loss:\t{}".format(_mapping_loss))
        print("\t>> Adversary Loss:\t{}".format(_adversary_loss))
        print("\t>> Domain Accuracy:\t{}".format(dom_acc))
        print("\t>> Source Accuracy:\t{}".format(src_acc))
        print("\t>> Target Accuracy:\t{}".format(tgt_acc))
        print("\t>> Generator Learning Rate:\t{}".format(lr_gen))
        print("\t>> Discriminator Learning Rate:\t{}".format(lr_gen))
        print("\t>> Executed Time:\t{} sec/iter".format(timer.toc()))

        if (iter % options.checkpoint_steps) == 0:
            target_saver.save(sess, target_model_path)
            adversary_saver.save(sess, adversary_model_path)
            print("Saving model at iteration {}".format(iter))


        if options.lr_decay_steps is not None and (iter % options.lr_decay_steps) == 0:
            current_learning_rate_generator = sess.run(learning_rate_generator_op.assign(
                                            current_learning_rate_generator * options.lr_decay_rate))
            current_learning_rate_discriminator = sess.run(learning_rate_discriminator_op.assign(
                                            current_learning_rate_discriminator * options.lr_decay_rate))

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == "__main__":
    main()



