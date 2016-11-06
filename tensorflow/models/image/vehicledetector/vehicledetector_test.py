

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import vehicledetector
import vehicledetector_input

import cv2

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_integer('batch_size', 128,
#                            """Number of images to process in a bat#ch""")

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/gauravgupta/workspace/mytensorflow/tensorflow/tensorflow/models/image/vehicledetector/trainedmodel', """Directory where model is saved""")

tf.app.flags.DEFINE_string('num_examples', 1, """Number of examples to run""")

def evaluate(file_path, y1, x1, h, w):

    with tf.Graph().as_default() as g:
        image = vehicledetector_input.read_image(file_path, y1, x1, h, w, FLAGS.batch_size)

        logits = vehicledetector.inference(image)
        _, top_k_pred = tf.nn.top_k(logits, k=1)

        variable_averages = tf.train.ExponentialMovingAverage(vehicledetector.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            answer = sess.run([top_k_pred])
            print ("answer=" + str(answer[0][0]))
                
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    cv_img = cv2.imread(file_path)
    cv2.rectangle(cv_img, (x1, y1), (x1+w, y1+h), (0,255,0), 1)
    cv2.imshow("output", cv_img)
    cv2.waitKey(0)

def main(argv):
    print ("vehicledetector test:" + argv[1])
    if not tf.gfile.Exists(argv[1]):
        print("File does not exist!")
        exit(0)
    x1 = int(argv[2])
    y1 = int(argv[3])
    w = int(argv[4])
    h = int(argv[5])
    evaluate(argv[1], y1, x1, h, w)
    
if __name__ == '__main__':
    tf.app.run()
