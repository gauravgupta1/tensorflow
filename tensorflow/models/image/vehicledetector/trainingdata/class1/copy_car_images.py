from subprocess import call
import tensorflow as tf

# python copy_car_images <file_path> <startnumber> <endnumber>

def copy_car_images(file_path, start_number, end_number):
    for x in xrange(start_number, end_number+1):
        call(["cp", "%s"%file_path, "%06d.jpg" % x])
    
def main(argv):
    file_path = argv[1]
    start_number = int(argv[2])
    end_number = int(argv[3])
    copy_car_images(file_path, start_number, end_number)
    
if __name__ == '__main__':
    tf.app.run()
