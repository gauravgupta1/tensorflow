
#USAGE: python vehicledetector_test.py <file_path.jpg> <x_start> <y_start> <ROI-width> <ROI-height>
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
        #_, top_k_pred = tf.nn.in_top_k(logits, k=1)

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
            logi, pred = sess.run([logits, top_k_pred])
            print(logi)    
            print ("answer=" + str(pred))
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return pred[0][0]

def main(argv):
    print ("vehicledetector test:" + argv[1])
    if not tf.gfile.Exists(argv[1]):
        print("File does not exist!")
        exit(0)
    file_path = argv[1]
    x1 = int(argv[2])
    y1 = int(argv[3])
    w = int(argv[4])
    h = int(argv[5])
    cv_img = cv2.imread(file_path)
    imgh,imgw = cv_img.shape[:2]

    for x in xrange(x1,(imgw*3)//4,w):
        for y in xrange(y1,(imgh*3)//4,h):
            print(y)
            retval = evaluate(file_path, y, x, h, w)
            if retval == 1:
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0,255,0), 1)
            else:
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0,0,255), 1)
            cv2.imshow("output", cv_img)
            cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.imwrite("testoutput.jpg", cv_img)
    
if __name__ == '__main__':
    tf.app.run()
    #main=main, argv=[sys.argv[0]])
