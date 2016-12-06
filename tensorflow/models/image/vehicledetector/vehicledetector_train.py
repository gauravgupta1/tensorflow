"""
USAGE: $ python vehicledetector_train.py

TRAINING DATA ORGANIZATION:
data_dir = /home/gaurav/workspace/car_ims
class0 = /home/gaurav/workspace/car_ims/class0/negative%06d.jpg
class1 = /home/gaurav/workspace/car_ims/class1/%06d.jpg
"""
#**********************************************************
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os.path
import numpy as np
from six.moves import xrange
import signal

import tensorflow as tf
import vehicledetector
import vehicledetector_input
import freeze_graph

#**********************************************************

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/vehicledetector_train',
                           """Directory where to write event logs """
                           """and checkpoint. """)
tf.app.flags.DEFINE_string('model_dir', '/tmp/vehicledetector_train', """Directory where model is saved """)
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                           """Number of batches to run. """)
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

IMAGE_SIZE_W = vehicledetector_input.IMAGE_SIZE_W

IMAGE_SIZE_H = vehicledetector_input.IMAGE_SIZE_H

_should_exit = False

input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"

#**********************************************************

def train():
    """Train vehicle detector for a number of steps """
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        # Get images and labels
        images_tensor, labels_tensor = vehicledetector.prepare_data(eval_data=False)
        
        placeholder_images = vehicledetector.placeholder_for_data(tf.float32, (FLAGS.batch_size, IMAGE_SIZE_H, IMAGE_SIZE_W, 3))
        placeholder_labels = vehicledetector.placeholder_for_data(tf.int64, (FLAGS.batch_size))
    
        # Build graph the computes the logits prediction
        # from the inference model
        logits = vehicledetector.inference(placeholder_images)

        # Calculate loss (cost function).
        loss = vehicledetector.loss(logits, placeholder_labels)

        ## Build a Graph that trains the model with one
        ## batch of examples and updates the model parameters.
        train_op = vehicledetector.train(loss, global_step)

        # Create a saver
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of summaries
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below
        init = tf.initialize_all_variables()

        # start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, inter_op_parallelism_threads=3, intra_op_parallelism_threads=3))
        sess.run(init)

        #start the queue runners
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        last_checkpoint_step = 0
        for step in xrange(FLAGS.max_steps):

            if _should_exit == True:
                break
            
            images, labels = sess.run([images_tensor, labels_tensor])
            #print(images.shape, labels.shape)
            #print(placeholder_images.get_shape(), placeholder_labels.get_shape())
            feed_dict = {placeholder_images: images, placeholder_labels: labels}
            
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            if loss_value > 100:
                print('Model diverged with loss = NaN')
            #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                last_checkpoint_step = step
                saver.save(sess, checkpoint_path, global_step=last_checkpoint_step)

        # save graph with last saved checkpoint file
        if last_checkpoint_step > 0:
            save_graph(sess, last_checkpoint_step)

#**********************************************************

def save_graph(sess, step):

    tf.train.write_graph(sess.graph.as_graph_def(), FLAGS.model_dir, input_graph_name)

    input_graph_path = os.path.join(FLAGS.model_dir, input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt-' + str(step))
    output_node_names = "softmax_linear"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(FLAGS.model_dir, output_graph_name)
    clear_devices = False

    freeze_graph.freeze_graph(input_graph_path,
                              input_saver_def_path,
                              input_binary,
                              input_checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_path,
                              clear_devices)
    
#**********************************************************

def signal_handler(signal, frame):
    print('You pressed Ctrl+C: Exiting...')
    global _should_exit
    _should_exit = True
    
def main(argv=None):
    signal.signal(signal.SIGINT, signal_handler)
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    if tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)
    train()

#**********************************************************

#USAGE: python vehicledetector_train.py    
if __name__ == '__main__':
    tf.app.run()

#**********************************************************
