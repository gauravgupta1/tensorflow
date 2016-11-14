
#USAGE: python vehicledetector_test.py <file_path.jpg> <x_start> <y_start> <ROI-width> <ROI-height>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

import vehicledetector
import vehicledetector_input
from tile import tile
from startposition import startposition
from pyramid import pyramid
import time

import cv2

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/vehicledetector_train', """Directory where model is saved""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/gauravgupta/workspace/mytensorflow/tensorflow/tensorflow/models/image/vehicledetector/trainedmodel', """Directory where model is saved""")

tf.app.flags.DEFINE_string('num_examples', 1, """Number of examples to run""")

def evaluate(cv_img, left_x, left_y, right_x, right_y, tile_h, tile_w, batch_size):

    with tf.Graph().as_default() as g:
        #image = vehicledetector_input.read_image(file_path, left_y, left_x, h, w, FLAGS.batch_size)

        image = vehicledetector_input.image2tensor(cv_img, left_x, left_y, right_x, right_y, tile_w, tile_h, batch_size)
        
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

def test_algo_tile_scan(cv_img, start_left_x, start_left_y, right_x, right_y, tile_w, tile_h, stride):
    detected_cars = []
    eval_img = cv_img.copy()
    imgh,imgw = cv_img.shape[:2]
    for (left_x, left_y) in startposition(start_left_x, start_left_y, right_x, right_y, tile_w, tile_h, stride):
        #print("left_x:%d,left_y%d"%(left_x,left_y))
        start_time = time.time()
        retval = evaluate(eval_img, left_x, left_y, right_x, right_y, tile_w, tile_h, batch_size=128)
        end_time = time.time()
        print("evaluate:time:%d"%(end_time-start_time))
        #cv_img = cv2.imread(file_path)

        index = 0
        for (x,y) in tile(left_x, left_y, right_x, right_y, tile_w, tile_h):
            if index >= 128:
                continue
            if retval[index] == 1:
                print('found')
                cv2.rectangle(cv_img, (x, y), (x+tile_w, y+tile_h), (0,255,0), 1)
                detected_cars.append([x,y])
            #else:
            #    cv2.rectangle(cv_img, (x, y), (x+tile_w, y+tile_h), (0,0,255), 1)
            cv2.imshow("output", cv_img)
            cv2.waitKey(1)
            index = index+1

    return detected_cars

def test_algo_sliding_window(cv_img, left_lane_percent, right_lane_percent, horizon_percent, car_dash_percent, tile_w, tile_h, stride):
    print("cv_img", cv_img.shape[1], cv_img.shape[0])
    y_top = (cv_img.shape[0] * horizon_percent) // 100
    y_bottom = (cv_img.shape[0] * car_dash_percent) // 100
    x_left = (cv_img.shape[1] * left_lane_percent) // 100
    x_right = (cv_img.shape[1] * right_lane_percent) // 100
    print("crop_dim", y_top, y_bottom, x_left, x_right)
    crop_img = cv_img[y_top:y_bottom, x_left:x_right]
    scale_factor = 2
    retval = []
    
    print("crop_img:",crop_img.shape[1], crop_img.shape[0])
    for resized,scale in pyramid(crop_img, scale_factor):
        print("resized",resized.shape[1], resized.shape[0])
        detected_cars = test_algo_tile_scan(resized, 0, 0, resized.shape[1], resized.shape[0], tile_w, tile_h, stride)
        for (x,y) in detected_cars:
            retval.append([x_left+int(x/scale), y_top+int(y/scale)])
            
    return retval

def main(argv):
    file_path = argv[1]
    print ("vehicledetector test:" + file_path)
    if not tf.gfile.Exists(file_path):
        print("File does not exist!")
        exit(0)
    left_lane_percent = int(argv[2])
    right_lane_percent = int(argv[3])
    horizon_percent = int(argv[4])
    car_dash_percent = int(argv[5])
    tile_w = int(argv[6])
    tile_h = int(argv[7])
    stride = int(argv[8])
    cv_img = cv2.imread(file_path)
    start_left_x = (left_lane_percent * cv_img.shape[1]) // 100
    start_left_y = (horizon_percent * cv_img.shape[0]) // 100
    right_x = (right_lane_percent * cv_img.shape[1]) // 100
    right_y = (car_dash_percent * cv_img.shape[0]) // 100
    
    #test_algo_tile_scan(cv_img, start_left_x, start_left_y, right_x, right_y, tile_w, tile_h, stride)

    start_time = time.time()
    retval = test_algo_sliding_window(cv_img, left_lane_percent, right_lane_percent, horizon_percent, car_dash_percent, tile_w, tile_h, stride)
    for (x,y) in retval:
        cv2.rectangle(cv_img, (x, y), (x+tile_w, y+tile_h), (0,255,0), 1)
    end_time = time.time()
    print("Total:time:%d"%(end_time-start_time))

    cv2.imshow("finaloutput", cv_img)
    cv2.waitKey(0)
    cv2.imwrite("testoutput.jpg", cv_img)
    
if __name__ == '__main__':
    # missed ATD
    tf.app.run()
    
