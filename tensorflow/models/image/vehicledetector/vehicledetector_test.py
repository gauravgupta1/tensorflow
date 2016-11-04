

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import vehicledetector
import vehicledetector_input

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_integer('batch_size', 128,
#                            """Number of images to process in a bat#ch""")

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/gauravgupta/workspace/mytensorflow/tensorflow/tensorflow/models/image/vehicledetector/trainedmodel', """Directory where model is saved""")

tf.app.flags.DEFINE_string('num_examples', 1, """Number of examples to run""")

def evaluate(file_path):

    with tf.Graph().as_default() as g:
        image = vehicledetector_input.read_image(file_path, FLAGS.batch_size)

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

def main(argv):
    print ("vehicledetector test:" + argv[1])
    if not tf.gfile.Exists(argv[1]):
        print("File does not exist!")
        exit(0)
    evaluate(argv[1])
    
if __name__ == '__main__':
    tf.app.run()
