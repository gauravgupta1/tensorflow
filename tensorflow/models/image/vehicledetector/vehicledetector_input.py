from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from scipy.io import loadmat
import scipy.misc

IMAGE_SIZE_W = 60
IMAGE_SIZE_H = 40

NUM_CLASSES = 2
# 2*10000 images (10000 each for classes: 0 and 1)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2000

class CARIMSRecord(object):
    pass

def print_tensor_info(t):
    print (str(t.name) + ": " + str(t.get_shape()) + " size:" + str(t.get_shape().ndims) + " dtype:" + str(t.dtype))

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

    whitened_image = tf.image.per_image_whitening(resized_image)
    print_tensor_info(whitened_image)


    result.label = tf.fill([1], class_label)
    result.floatimage = whitened_image

    return result
    

def _generate_image_and_label_batch(data_list, min_queue_examples, batch_size, shuffle):
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
    if shuffle:
        images, label_batch = tf.train.shuffle_batch_join(
            data_list,
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch_join(
            data_list,
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size)

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
    lowrange = 0
    highrange = 1

    if not eval_data:
        lowrange = 1
        highrange = 10000
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        lowrange = 10001
        highrange = 11000
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        
    del filenames[:]
    for i in range(lowrange, highrange):
        filename = matfile['annotations'][i-1].relative_im_path
        filename = filename.replace("car_ims", "")
        filename = data_dir + filename
        filenames.append(filename)
    filename_queue = tf.train.string_input_producer(filenames)
    read_input_vehicle = read_car_ims(filename_queue, 1, eval_data)

    del filenames[:]
    for i in range(lowrange, highrange):
        filename = data_dir + "/class0/negative" + str(i).zfill(6) + ".jpg"
        filenames.append(filename)
    filename_queue = tf.train.string_input_producer(filenames)
    read_input_no_vehicle = read_car_ims(filename_queue, 0, eval_data)

    datalist = [(read_input_vehicle.floatimage, read_input_vehicle.label), (read_input_no_vehicle.floatimage, read_input_no_vehicle.label)] 

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    images, labels = _generate_image_and_label_batch(datalist, min_queue_examples, batch_size, shuffle=False)               

    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # with sess as sess:
    #     sess.run(init)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     img, label = sess.run([images, labels])
    #     for i in xrange(10):
    #         name = os.path.join('/tmp/test', str(i) +'.jpeg')
    #         print(label[i])
    #         print(name)
    #         scipy.misc.imsave(name, img[i])
    #     coord.request_stop()
    #     coord.join(threads)
    # print('written')

    return images, labels
        
    
    
          
