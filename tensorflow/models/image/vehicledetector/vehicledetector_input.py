from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from scipy.io import loadmat
import scipy.misc
import cv2
from tile import tile
from startposition import startposition
import time
import threading
import concurrent.futures
import Queue
from multiprocessing.pool import ThreadPool

IMAGE_SIZE_W = 60
IMAGE_SIZE_H = 40

NUM_CLASSES = 2
# 2*10000 images (10000 each for classes: 0 and 1)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 140000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2000

class CARIMSRecord(object):
    pass

def print_tensor_info(t):
    print (str(t.name) + ": " + str(t.get_shape()) + " size:" + str(t.get_shape().ndims) + " dtype:" + str(t.dtype))

def dump_batch_images(image_tensor, batch_size):
    init = tf.initialize_all_variables()
    sess = tf.Session()
    with sess as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        imgs = sess.run(image_tensor)
        for i in xrange(batch_size):
            name = os.path.join('/tmp/test', str(i) +'.jpeg')
            #print(name)
            #print(imgs[i].shape)
            scipy.misc.imsave(name, imgs[i])
        coord.request_stop()
        coord.join(threads)
    print('batch dumped to /tmp/test')


def read_car_ims(filename_queue, class_label, eval_data=False):
    """Reads and parses images from car_ims image files.

    Recommendation: @TODO parrallelism is possible

    Args:
      filename_queue: path for one file in car_ims
    
    Returns:
      An object representing a single image, with the following fields:
          height: number of rows in the result (H)
          width: number of rows in the result (W)
          depth: number of color channels in the result (3)
          key: a scalar string Tensor describing the file & record number for this example.
          label: an int32 Tensor with the label in the range 0,1
          floatimage: a [height, width, depth] uint8 Tensor with the image data
    """

    result = CARIMSRecord()

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    orig_image = tf.image.decode_jpeg(value, channels=3)
    #orig_image = tf.image.decode_png(value, channels=3)
        
    resized_image = tf.image.resize_images(orig_image, [IMAGE_SIZE_H, IMAGE_SIZE_W], 0, False)
    print_tensor_info(resized_image)

    #whitened_image = resized_image
    whitened_image = tf.image.per_image_whitening(resized_image)
    #float_image =    tf.image.per_image_standardization(resized_image)
    #whitened_image = tf.image.per_image_standardization(resized_image)
    print_tensor_info(whitened_image)


    result.label = tf.fill([1], class_label)
    result.floatimage = whitened_image

    return result

def read_image_bounding_box(filename_queue, y1, x1, height, width):
    """Reads and parses image from a jpg.

    Recommendation: @TODO parrallelism is possible

    Args:
      filename_queue: path for one file
      y1
      x1
      height
      width
    
    Returns:
      Single image in float
    """

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    orig_image = tf.image.decode_jpeg(value, channels=3)
    #orig_image = tf.image.decode_png(value, channels=3)

    cropped_image = tf.image.crop_to_bounding_box(orig_image, y1, x1, height, width)
    resized_image = tf.image.resize_images(cropped_image, [IMAGE_SIZE_H, IMAGE_SIZE_W], 0, False)
    print_tensor_info(resized_image)

    #whitened_image = resized_image
    whitened_image = tf.image.per_image_whitening(resized_image)
    print_tensor_info(whitened_image)

    float_image = whitened_image

    return float_image


def _generate_image_and_label_batch(data_list, min_queue_examples, batch_size, shuffle, eval_data):
    """Construct a queued batch of images and labels.
    
    Args:
      image: 3D Tensor of  [height, width, 3] of type float32
      label: 1D Tensor of type.int32
      min_queue_example: int32, minimum number of samples to retain in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    """
    num_preprocess_threads = 16
    
    if not eval_data:
        vehicle_batch_size = batch_size // 4
        non_vehicle_batch_size = (batch_size*3) // 4
    else:
        vehicle_batch_size = batch_size
        non_vehicle_batch_size = 0
        
    images0, label_batch0 = tf.train.batch(
        data_list[0],
        batch_size=vehicle_batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * vehicle_batch_size)
    if not eval_data:
        images1, label_batch1 = tf.train.batch(
            data_list[1],
            batch_size=non_vehicle_batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * non_vehicle_batch_size)
        images = tf.concat(0, [images0, images1])
        label_batch = tf.concat(0, [label_batch0, label_batch1])
    else:
        images = images0
        label_batch = label_batch0

    # images, label_batch = tf.train.batch(
    #     data_list,
    #     batch_size=batch_size,
    #     enqueue_many=True,
    #     num_threads=num_preprocess_threads,
    #     capacity=min_queue_examples + 3 * vehicle_batch_size)
        
    if shuffle:
        images = tf.random_shuffle(images)
        label_batch = tf.random_shuffle(label_batch)

    print(images.get_shape())
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])
        
        
def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CARIMS training using the Reader ops.
    
    Args:
      data_dir: Path to CAR_IMS data directory
      batch_size: Number of images per batch.
    
    Returns:
      images: Images: 4D tensor of [batch_size, IMAGE_SIZE_W, IMAGE_SIZE_H, 3] size.
      labels: Lables. 1D tensor of [batch_size] size.
    """

def inputs(eval_data, data_dir, batch_size):
    """Construct input for VEHICLEDETECTOR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the car_ims directory
      batch_size: Number of images per batch
      
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, iMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    """
    matfile = loadmat(data_dir + "/cars_annos.mat", squeeze_me=True, struct_as_record=False)
    type(matfile)
    filenames = []
    filenames1 = []
    lowrange = 0
    highrange = 1

    if not eval_data:
        lowrange_class0 = 1
        highrange_class0 = 100000
        lowrange_class1 = 1
        highrange_class1 = 40000
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        lowrange_class0 = 79500
        highrange_class0 = 80500
        lowrange_class1 = 30000
        highrange_class1 = 31000
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # negative training data
    del filenames[:]
    for i in range(lowrange_class0, highrange_class0):
        filename = data_dir + "/class0/negative" + str(i).zfill(6) + ".jpg"
        filenames.append(filename)
    filename_queue = tf.train.string_input_producer(filenames)
    read_input_no_vehicle = read_car_ims(filename_queue, 0, eval_data)
    
    # positive training data
    for i in range(lowrange_class1, highrange_class1):
        filename = data_dir + "/class1/" + str(i).zfill(6) + ".jpg"
        filenames1.append(filename)
    filename_queue1 = tf.train.string_input_producer(filenames1)
    read_input_vehicle = read_car_ims(filename_queue1, 1, eval_data)

    #datalist = [(read_input_no_vehicle.floatimage, read_input_no_vehicle.label), (read_input_vehicle.floatimage, read_input_vehicle.label)] 
    datalist = [(read_input_vehicle.floatimage, read_input_vehicle.label), (read_input_no_vehicle.floatimage, read_input_no_vehicle.label)] 

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    images, labels = _generate_image_and_label_batch(datalist, min_queue_examples, batch_size, shuffle=False, eval_data=eval_data)               

    dump_batch_images(images, batch_size)

    return images, labels

def tile_image(q, graph, float_image, left_x, left_y, right_x, right_y, tile_width, tile_height, image_width, image_height):
    print("tile_image:tid:", threading.current_thread().ident)
    with graph.as_default():
        first_tile = True

        for (x,y) in tile(left_x, left_y, right_x, right_y, tile_width, tile_height):
            if y+tile_height > image_height or x+tile_width > image_width:
                continue
        
            #print (x,y,float_image.get_shape())
            img_slice = tf.slice(float_image, [y,x,0], [tile_height, tile_width, 3])
            if tile_height != IMAGE_SIZE_H or tile_width != IMAGE_SIZE_W:
                img_slice = tf.image.resize_images(img_slice, [IMAGE_SIZE_H, IMAGE_SIZE_W], 0, False)
            img_slice = tf.image.per_image_whitening(img_slice)
            img_slice = tf.expand_dims(img_slice, 0)
            if first_tile:
                first_tile = False
                image_tensor = img_slice
                x_tensor = tf.fill([1], x)
                y_tensor = tf.fill([1], y)
            else:
                image_tensor = tf.concat(0, [image_tensor, img_slice])
                x_tensor = tf.concat(0,[x_tensor, tf.fill([1], x)])
                y_tensor = tf.concat(0,[y_tensor, tf.fill([1], y)])
                
        q.put((image_tensor, x_tensor, y_tensor))
        return (image_tensor, x_tensor, y_tensor)

def image2tensor(graph, cv_img, start_left_x, start_left_y, right_x, right_y, tile_width, tile_height, stride, batch_size):

    orig_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    float_image = tf.cast(orig_image, tf.float32)
    image_height = orig_image.shape[0]
    image_width = orig_image.shape[1]
    t_list = []
    total_tile_count = 0
    
    begin_time = time.time()
    list_lock = threading.Lock()
    jobs = []
    q = Queue.Queue()
    pool = ThreadPool(processes=10)

    for (left_x, left_y) in startposition(start_left_x, start_left_y, right_x, right_y, tile_width, tile_height, stride):
        async_result = pool.apply_async(tile_image, (q, graph, float_image, left_x, left_y, right_x, right_y, tile_width, tile_height, image_width, image_height))
        jobs.append(async_result)
        
    for t in jobs:
        (x, y, z) = t.get()
        t_list.append((x,y,z))
        total_tile_count += x.get_shape()[0].value
        print("total_tile_count:",total_tile_count)
        
    # threading.thread
    # for (left_x, left_y) in startposition(start_left_x, start_left_y, right_x, right_y, tile_width, tile_height, stride):
    #     t = threading.Thread(target=tile_image, args=(q, graph, float_image, left_x, left_y, right_x, right_y, tile_width, tile_height, image_width, image_height))
    #     jobs.append(t)
    #     t.daemon = True
    #     t.start()

    # for t in jobs:
    #     t.join()

    # while not q.empty():
    #     (x, y, z) = q.get()
    #     t_list.append((x,y,z))
    #     total_tile_count += x.get_shape()[0].value
        
    # with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    #     for (left_x, left_y) in startposition(start_left_x, start_left_y, right_x, right_y, tile_width, tile_height, stride):
    #         jobs.append(executor.submit(tile_image, q, graph, float_image, left_x, left_y, right_x, right_y, tile_width, tile_height, image_width, image_height))
            
    #     try:
    #         for future in concurrent.futures.as_completed(jobs):
    #             #list_lock.acquire()
    #             (x,y,z) = future.result()
    #             t_list.append((x,y,z))
    #             total_tile_count += x.get_shape()[0].value
    #             print("total_tile_count:",total_tile_count)
    #             #list_lock.release()
    #     except Exception as e:
    #         print("Exception:%s" % (e))

    # All sequential - fresh reboot (43 secs)
    # for (left_x, left_y) in startposition(start_left_x, start_left_y, right_x, right_y, tile_width, tile_height, stride):
    #     (x, y, z) = tile_image(q, graph, float_image, left_x, left_y, right_x, right_y, tile_width, tile_height, image_width, image_height)
            
    #     t_list.append((x,y,z))
    #     total_tile_count += x.get_shape()[0].value
    #     print("total_tile_count:",total_tile_count)
            
            
    # if less than batch_size, fill with last    
    # batch_count = image_tensor.get_shape()[0]
    # batch_count = len(t_list)
    # filler_count = batch_size - (batch_count%batch_size)
    # print(filler_count)
    # for b in xrange(0, filler_count, 1):
    #     #image_tensor = tf.concat(0, [image_tensor, img_slice])
    #     t_list[len(t_list)-1].append((img_slice,tf.fill([1],x),tf.fill([1]
                                                                       # ,y)))

    end_time = time.time()
    print("tiling-delay:%d"%(end_time-begin_time))
    
    print("len(t_list)", len(t_list))

    begin_time = time.time()
    retval,x,y = tf.train.batch_join(t_list, batch_size=batch_size, enqueue_many=True)
    end_time = time.time()
    print("batching-delay:%d"%(end_time-begin_time))

    #print(len(retval))
    #print_tensor_info(retval)
    dump_batch_images(retval, batch_size)
    #print_tensor_info(x)
    steps = total_tile_count // batch_size
    
    return retval,x,y,steps
    
def read_image(file_path, y1, x1, height, width, batch_size):

    """ 
    Input:
    file_path: must be a jpg image (not png)

    """
    filenames = [file_path]
    filename_queue = tf.train.string_input_producer(filenames)
    image_data = read_image_bounding_box(filename_queue, y1, x1, height, width)

    #temp = [image_data]
    #image_data = tf.train.batch(temp, batch_size=batch_size)
    image_data = tf.expand_dims(image_data, 0)
    image_data = tf.tile(image_data, tf.pack([batch_size, 1, 1, 1]))
    print (image_data.get_shape())

    dump_batch_images(image_data, batch_size)

    return image_data
