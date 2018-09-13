# coding:utf-8

from digit_config import *
import digit_model
import tensorflow as tf
import csv


def digit_eval_input():
    # 构造输入数据流
    filename = ['data/test.csv']
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=11430)
    key, value = reader.read(filename_queue)
    record_defaults = [[0] for _ in range(784)]
    record_bytes = tf.decode_csv(value, record_defaults=record_defaults)
    image_value_temp = tf.slice(record_bytes, [0], [784])
    image_value = tf.reshape(image_value_temp, [28, 28])
    image_batch = tf.train.batch([image_value], batch_size=batch_size, num_threads=1, capacity=200)
    image_batch = tf.reshape(image_batch, [batch_size, 28, 28, 1])
    return tf.div(tf.cast(image_batch, tf.float32), 255.0)


def main():
    images_test = digit_eval_input()
    label_predict = digit_model.inference(images_test, keep_prob=1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess=sess, save_path='log/model.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        f = open('data/test_result.csv', 'a', newline='')
        csv_write = csv.writer(f, dialect='excel')
        num = 11430
        while True:
            label_test = sess.run(tf.argmax(label_predict, 1))
            i = 0
            for item in label_test:
                csv_write.writerow([num+i, item])
                print(num+i, item)
                i += 1
            num += 1
            if num > 28001:
                break
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
