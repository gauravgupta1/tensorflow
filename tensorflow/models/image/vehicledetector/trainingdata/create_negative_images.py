from subprocess import call
import tensorflow as tf

def main(argv):
    start_number = 1
    for x in xrange(0,1920,60):
        for y in xrange(0,1080,40):
            call(["ffmpeg", "-loglevel", "panic", "-i", "/mnt/hgfs/shared/2015-01-01-10-45-12.MOV", "-r", "1/5", "-vf", "crop=x=%d:y=%d:w=60:h=40" % (x,y), "-start_number", "%d" % start_number, "-q:v", "2", "negative%06d.jpg"])
            start_number = start_number+(60*1/5)
            print(start_number)

if __name__ == '__main__':
    tf.app.run()
