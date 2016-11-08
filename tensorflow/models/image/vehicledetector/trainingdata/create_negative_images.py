from subprocess import call
import tensorflow as tf

def main(argv):

    # verify
    # for counter in xrange(1,100000):
    #     if not tf.gfile.Exists("negative%06d.jpg" % counter):
    #         print("file does not exist:" + "negative%06d.jpg" % counter)
        
    # return
    
    start_number = int(argv[2])
    for x in xrange(0,1920,60):
        for y in xrange(0,1080,40):
            call(["ffmpeg", "-loglevel", "panic", "-i", argv[1], "-r", "1/5", "-vf", "crop=x=%d:y=%d:w=60:h=40" % (x,y), "-start_number", "%d" % start_number, "-q:v", "2", "negative%06d.jpg"])
            start_number = start_number+(60*30)
            print(start_number)

if __name__ == '__main__':
    tf.app.run()
