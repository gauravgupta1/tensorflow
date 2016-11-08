from subprocess import call
import tensorflow as tf

# python copy_negative_images <file_path> <startnumber> <endnumber>
def main(argv):
    file_path = argv[1]
    start_number = int(argv[2])
    end_number = int(argv[3])
    for x in xrange(start_number, end_number+1):
        call(["cp", "%s"%file_path, "negative%06d.jpg" % x])

if __name__ == '__main__':
    tf.app.run()
