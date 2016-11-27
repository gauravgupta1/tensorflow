####
#### Usage: python vehicledetector_test.py <input_file> <left_lane_percent> <right_lane_percent> <horizon_percent> <car_dash_percent> <tile_width> <tile_height> <stride>
####
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import numpy as np
import tensorflow as tf

import vehicledetector
import vehicledetector_input
from tile import tile
from startposition import startposition
from pyramid import pyramid
import time

from multiprocessing import Process, Queue, Pool

import cv2

IMAGE_SIZE_W = vehicledetector_input.IMAGE_SIZE_W
IMAGE_SIZE_H = vehicledetector_input.IMAGE_SIZE_H
FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/vehicledetector_train', """Directory where model is saved""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/gaurav/workspace/tensorflow/tensorflow/models/image/vehicledetector/trainedmodel', """Directory where model is saved""")

tf.app.flags.DEFINE_string('num_examples', 1, """Number of examples to run""")

def get_time_in_ms():
    return int(round(time.time()*1000))

def evaluate(cv_img, start_left_x, start_left_y, right_x, right_y, tile_h, tile_w, stride, batch_size):

    with tf.Graph().as_default() as g:

        images,x,y,steps = vehicledetector_input.image2tensor(g, cv_img, start_left_x, start_left_y, right_x, right_y, tile_w, tile_h, stride, batch_size)
        
        logits = vehicledetector.inference(images)
        _, top_k_pred = tf.nn.top_k(logits, k=1)

        variable_averages = tf.train.ExponentialMovingAverage(vehicledetector.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                    if len(threads) >= 16:
                        break
                    
                #threads = tf.train.start_queue_runners(coord=coord)
                print("len(threads)",len(threads))
                detected_cars = {}
                for step in xrange(0,steps):
                    x1, y1, logi, pred = sess.run([x, y, logits, top_k_pred])
                    for index in xrange(0,len(pred)):
                        if pred[index] == 1:
                            detected_cars[x1[index]] = y1[index]
                            print (step, x1[index], y1[index])
            except Exception as e:
                coord.request_stop(e)
                
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return detected_cars

def evaluate2(cv_image_list, x_list, y_list, batch_size):

    with tf.Graph().as_default() as g:

        placeholder_images = vehicledetector.placeholder_for_data(tf.float32, (batch_size, IMAGE_SIZE_H, IMAGE_SIZE_W, 3))

        logits = vehicledetector.inference(placeholder_images)
        _, top_k_pred = tf.nn.top_k(logits, k=1)

        variable_averages = tf.train.ExponentialMovingAverage(vehicledetector.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # image_queue = tf.train.input_producer(cv_image_list, element_shape=[40, 60, 3])
        # x_queue = tf.train.input_producer(x_list)
        # y_queue = tf.train.input_producer(y_list)
        steps = len(cv_image_list) // batch_size

        #config = tf.ConfigProto(device_count = {'GPU': 0})
        #with tf.Session(config=config) as sess:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('loading ckpt model from:',ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # start the queue runners.
            coord = tf.train.Coordinator()
            detected_cars = {}
            try:
                threads = []
                threads = tf.train.start_queue_runners(coord=coord)
                print("len(threads)",len(threads))

                start_time = get_time_in_ms()

                # for step in xrange(0, steps):
                #     image_batch_t = image_queue.dequeue_many(batch_size)
                #     x_batch_t = x_queue.dequeue_many(batch_size)
                #     y_batch_t = y_queue.dequeue_many(batch_size)
                    
                #     image_batch, x_batch, y_batch = sess.run([image_batch_t, x_batch_t, y_batch_t])
                #     feed_dict = {placeholder_images: image_batch}
                    
                #     logi, pred = sess.run([logits, top_k_pred], feed_dict=feed_dict)
                #     for index in xrange(0,len(pred)):
                #         if pred[index] == 1:
                #             detected_cars[x_batch[index]] = y_batch[index]
                #             print (x_batch[index], y_batch[index])
                
                steps = len(cv_image_list) // batch_size
                for step in xrange(0,steps):
                    image_batch = cv_image_list[step*batch_size:(step+1)*batch_size]
                    
                    feed_dict = {placeholder_images: image_batch}
                    
                    logi, pred = sess.run([logits, top_k_pred], feed_dict=feed_dict)
                    for index in xrange(0,len(pred)):
                        if pred[index] == 1:
                            detected_cars[x_list[(step*batch_size)+index]] = y_list[(step*batch_size)+index]
                            print (step, x_list[(step*batch_size)+index], y_list[(step*batch_size)+index])

                end_time = get_time_in_ms()
                print("tf.session:time:%dms"%(end_time-start_time))
            
            except Exception as e:
                coord.request_stop(e)
                
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return detected_cars

def tile_process(cv_img, left_x, left_y, right_x, right_y, tile_w, tile_h):
    retval_images = []
    for (x,y) in tile(left_x, left_y, right_x, right_y, tile_w, tile_h):
        #print(x,y)
        retval_images.append((per_image_whitening(cv_img[y:y+tile_h, x:x+tile_w]), x, y))

    return retval_images

def per_image_whitening(cv_img):
    num_pixels = cv_img.shape[0] * cv_img.shape[1]
    image_mean = np.average(cv_img)

    variance = np.average(np.square(cv_img)) - np.square(image_mean)
    variance = np.maximum(0, variance)
    stddev = np.sqrt(variance)
    
    min_stddev = 1 / np.sqrt(num_pixels)

    pixel_value_scale = np.maximum(stddev, min_stddev)
    pixel_value_offset = image_mean

    cv_img = np.subtract(cv_img, pixel_value_offset)
    cv_img = np.divide(cv_img, pixel_value_scale)

    return cv_img
    
def test_algo_tile_scan(cv_img, start_left_x, start_left_y, right_x, right_y, tile_w, tile_h, stride):
    detected_cars = {}
    eval_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    eval_img = eval_img.astype('float32')
    #eval_img = per_image_whitening(eval_img)
    print(eval_img.dtype)
    
    #detected_cars = evaluate(eval_img, start_left_x, start_left_y, right_x, right_y, tile_w, tile_h, stride, batch_size=128)
    #len(detected_cars)
    #detected_cars.update(retval)

    all_results = []
    def log_results(results):
        all_results.extend(results)
        
    start_time = get_time_in_ms()

    all_tiles = []
    jobs = []
    queue = Queue()
    index = 0
    max_processes = 10
    process_pool = Pool(max_processes)
    for (left_x,left_y) in startposition(start_left_x, start_left_y, right_x, right_y, tile_w, tile_h, stride):
        # launch a process to create tiles
        process_pool.apply_async(tile_process, args=(eval_img, left_x, left_y, right_x, right_y, tile_w, tile_h), callback=log_results)
        index += 1
    process_pool.close()
    process_pool.join()

    cv_img_list = []
    x = []
    y = []
    index = 0
    for tuple in all_results:
        #img = tuple[0].tolist()
        img = tuple[0]
        cv_img_list.append(img)
        cv2.imwrite("/tmp/test/cvimg%d.jpg"%(index), tuple[0])
        index += 1
        x.append(tuple[1])
        y.append(tuple[2])

    end_time = get_time_in_ms()
    print("tiling_process:time:%dms"%(end_time-start_time))
    print("cv_img_list:", len(cv_img_list), len(x), len(y))
    detected_cars = evaluate2(cv_img_list, x, y, batch_size=128)

    return detected_cars

def test_algo_sliding_window(cv_img, left_lane_percent, right_lane_percent, horizon_percent, car_dash_percent, tile_w, tile_h, stride):
    print("cv_img", cv_img.shape[1], cv_img.shape[0])
    y_top = (cv_img.shape[0] * horizon_percent) // 100
    y_bottom = (cv_img.shape[0] * car_dash_percent) // 100
    x_left = (cv_img.shape[1] * left_lane_percent) // 100
    x_right = (cv_img.shape[1] * right_lane_percent) // 100
    print("crop_dimensions:", y_top, y_bottom, x_left, x_right)
    crop_img = cv_img[y_top:y_bottom, x_left:x_right]
    scale_factor = 2
    retval = []
    
    for resized, scale in pyramid(crop_img, scale_factor):
        print("resized-image-shape:",resized.shape[1], resized.shape[0])
        detected_cars = test_algo_tile_scan(resized, 0, 0, resized.shape[1], resized.shape[0], tile_w, tile_h, stride)
        for x in detected_cars:
            y = detected_cars[x]
            retval.append([x_left+int(x/scale), y_top+int(y/scale)])
            
    return retval

def main(argv):
    if len(argv) < 9:
        print("Usage: python vehicledetector_test.py <input_file> <left_lane_percent> <right_lane_percent> <horizon_percent> <car_dash_percent> <tile_width> <tile_height> <stride>")
        exit(0)

    # read arguments
    file_path = argv[1]
    print ("vehicledetector_test: input file path:" + file_path)
    
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

    start_time = get_time_in_ms()
    detected_cars = test_algo_sliding_window(cv_img, left_lane_percent, right_lane_percent, horizon_percent, car_dash_percent, tile_w, tile_h, stride)
    for (x,y) in detected_cars:
        cv2.rectangle(cv_img, (x, y), (x+tile_w, y+tile_h), (0,255,0), 1)
    end_time = get_time_in_ms()
    print("Total:time:%dms"%(end_time-start_time))

    cv2.imshow("finaloutput", cv_img)
    cv2.waitKey(0)
    cv2.imwrite("testoutput.jpg", cv_img)
    
if __name__ == '__main__':
    # missed ATD
    tf.app.run()
    
