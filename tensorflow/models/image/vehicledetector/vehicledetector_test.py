
#USAGE: python vehicledetector_test.py <file_path.jpg> <x_start> <y_start> <ROI-width> <ROI-height>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import vehicledetector
import vehicledetector_input
from tile import tile

import cv2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/vehicledetector_train', """Directory where model is saved""")

tf.app.flags.DEFINE_string('num_examples', 1, """Number of examples to run""")

def evaluate(file_path, left_y, left_x, imgh, imgw, h, w, batch_size):

    with tf.Graph().as_default() as g:
        #image = vehicledetector_input.read_image(file_path, left_y, left_x, h, w, FLAGS.batch_size)

        image = vehicledetector_input.read_image2(file_path, left_y, left_x, (imgh*3)//4, (imgw*3)//4, h, w, batch_size)
        
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
            logi, pred = sess.run([logits, top_k_pred])
            #print(logi)    
            #print ("answer=" + str(pred))
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return pred
            #return pred[0][0]

def main(argv):
    print ("vehicledetector test:" + argv[1])
    if not tf.gfile.Exists(argv[1]):
        print("File does not exist!")
        exit(0)
    file_path = argv[1]
    left_x = int(argv[2])
    left_y = int(argv[3])
    w = int(argv[4])
    h = int(argv[5])

    cv_img = cv2.imread(file_path)
    imgh,imgw = cv_img.shape[:2]
    
    retval = evaluate(file_path, left_y, left_x, imgh, imgw, h, w, 128)

    index = 0
    for (x,y) in tile(left_x, left_y, (imgw*3)//4, (imgh*3)//4, w, h):
        if index >= 128:
            continue
        #retval = evaluate(file_path, y, x, imgh, imgw, h, w, 128)
        if retval[index] == 1:
        #if retval == 1:
            cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0,255,0), 1)
        else:
            cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0,0,255), 1)
            cv2.imshow("output", cv_img)
            cv2.waitKey(1)
        index = index+1

    cv2.waitKey(0)
    cv2.imwrite("testoutput.jpg", cv_img)
    
if __name__ == '__main__':
    tf.app.run()
    #main=main, argv=[sys.argv[0]])
