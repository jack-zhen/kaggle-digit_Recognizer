# coding: utf-8

import tensorflow as tf
from digit_config import *


def digit_input(filename=['data/train.csv']):
    # 构造输入数据流
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    record_defaults = [[0] for i in range(785)]
    record_bytes = tf.decode_csv(value, record_defaults=record_defaults)
    image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
    image_extracted = tf.slice(record_bytes, [1], [784])
    image_value = tf.reshape(image_extracted, [28, 28])
    image_batch, label_batch = tf.train.batch([image_value, image_label], batch_size=batch_size, num_threads=3,
                                              capacity=100 + batch_size*3)
    image_batch = tf.reshape(image_batch, [batch_size, 28, 28, 1])
    return tf.div(tf.cast(image_batch,tf.float32), 255.0), label_batch


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name,mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar(name,stddev)


def biases_generator(name, shape, initializer):
    biases = tf.get_variable(name, shape=shape, initializer=initializer)
    return biases


def weights_generator(name, shape, initializer):
    weights = tf.get_variable(name, shape=shape, initializer=initializer)
    return weights


def inference(images, keep_prob=0.5):
    images = tf.cast(images, tf.float32)
    # 第一层巻积
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', shape=[5, 5, 1, 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05),
                                 dtype=tf.float32)
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.))
        conv1_temp = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1_temp, biases), name=scope.name)
        print("After  conv1 data shape",conv1.shape)
        variable_summaries(kernel, 'conv1/kernel')
        variable_summaries(biases, 'conv1/biases')
    # 池化层1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    print("After  pool1 data shape", pool1.shape)

    # 第二层巻积
    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape=[5, 5, 64, 128],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05),
                                 dtype=tf.float32)
        conv2_temp = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2_temp, biases))
        variable_summaries(kernel, 'conv2/kernel')
        variable_summaries(biases, 'conv2/biases')
        print("After  conv2 data shape", conv2.shape)
    # 池化层2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print("After  pool2 data shape", pool2.shape)

    # 全链接层1
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.shape[1].value
        weights = weights_generator('weights3', [dim, 500],
                                    tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
        biases = biases_generator('biases', [500], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)
        local3_drop = tf.nn.dropout(local3, keep_prob=keep_prob)
        print("After local3 data shape", local3.shape)
        variable_summaries(weights, 'local3/weights')
        variable_summaries(biases, 'local3/biases')

    # 全链接层2
    with tf.variable_scope('local4') as scope:
        weights = weights_generator('weights4', [500, 250],
                                    tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
        biases = biases_generator('biases', [250], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3_drop, weights)+biases, name=scope.name)
        variable_summaries(weights, 'local4/weights')
        variable_summaries(biases, 'local4/biases')
        print("After local4 data shape", local4.shape)

    # Softmax 分类层
    with tf.variable_scope('softmax_linear') as scope:
        weights = weights_generator('weights', [250, 10],
                                    tf.truncated_normal_initializer(stddev=1/250.0, dtype=tf.float32))
        biases = biases_generator('biases', [10], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        print("After softmax_linear data shape", softmax_linear.shape)
        variable_summaries(weights, 'softmax/weights')
        variable_summaries(biases, 'softmax/biases')
    return softmax_linear


def loss(logits, label):
    label = tf.cast(label, tf.int64)
    label = tf.reshape(label,[batch_size])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    writer = tf.summary.FileWriter("./log", tf.get_default_graph())
    writer.close()
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
