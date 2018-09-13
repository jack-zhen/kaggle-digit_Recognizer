# coding:utf-8
# 模型数据输入模块

import tensorflow as tf
import digit_model
from digit_config import *


with tf.Graph().as_default():
    image_batch, label_batch = digit_model.digit_input()
    logits = digit_model.inference(image_batch)
    loss = digit_model.loss(logits, label_batch)
    global_step = tf.Variable(0, trainable=False)
    decay_learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                            decay_step, decay_rate, staircase=True,
                                                     name='decay_learning_rate')
    with tf.name_scope('decay_learning_rate'):
        tf.summary.scalar("learning_rate", decay_learning_rate)
    train_op = tf.train.AdamOptimizer(decay_learning_rate).minimize(loss, global_step=global_step)

    label_batch = tf.cast(tf.reshape(label_batch, [batch_size]), tf.int64)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), label_batch)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    # 神经网络搭建
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        images, labels = digit_model.digit_input()
        for i in range(train_step):
            summary, _ = sess.run([merged, train_op])
            summary_writer.add_summary(summary, i)
            if i % 30 == 0 or i+1 == train_step:
                saver.save(sess, 'log/model.ckpt',global_step=i)
        summary_writer.close()
        coord.request_stop()
        coord.join(threads)
