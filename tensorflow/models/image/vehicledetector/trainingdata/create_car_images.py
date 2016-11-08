from subprocess import call
import tensorflow as tf
import cv2
from copy_car_images import copy_car_images

# USAGE: python create_car_images.py <directory_path> <read_start_number> <read_end_number> <write_file_start_number> <write_file_number_increment>
def main(argv):

    for counter in xrange(1,40000):
        if not tf.gfile.Exists("%06d.jpg" % counter):
            print("file does not exist:" + "%06d.jpg" % counter)
    return

    directory = argv[1]
    start_number = int(argv[2])
    end_number = int(argv[3])
    output_file_counter_start = int(argv[4])
    output_file_counter_increment = int(argv[5])
    auto_mode = True
    counter = start_number
    
    while counter < end_number:
        file_path = directory + "/%06d.jpg"%counter
        cvimg = cv2.imread(file_path)
        cv2.imshow(file_path, cvimg)
        cv2.moveWindow(file_path, 0, 0)
        if auto_mode:
            wait_time = 500
        else:
            wait_time = 0
        key = cv2.waitKey(wait_time)
        # pause
        if key == 112:
            auto_mode = False
            key = cv2.waitKey(0)
        # next
        if key == 110:
            print('forward')
        # back
        if key == 98:
            counter = counter-1
            cv2.destroyAllWindows()
            continue
        # continue automode
        if key == 99:
            auto_mode = True
        if key == 121:
            copy_car_images(file_path, output_file_counter_start, output_file_counter_start+output_file_counter_increment)
            output_file_counter_start = output_file_counter_start+output_file_counter_increment
            print("new pos:" + str(output_file_counter_start))
        cv2.destroyAllWindows()
        counter = counter+1
        
if __name__ == '__main__':
    tf.app.run()
