from scipy.io import loadmat
import cv2

def main(argv=None):
    data_dir = '/home/gauravgupta/workspace/car_ims'
    matfile = loadmat(data_dir + '/cars_annos.mat', squeeze_me=True, struct_as_record=False)
    
    for i in range(101, matfile['annotations'].size+1):
    #for i in range(11, 100):
        filename = matfile['annotations'][i-1].relative_im_path
        filename = filename.replace("car_ims", "")
        filename = data_dir + filename
        print ( str(i) + " " + filename)
        x1 = matfile['annotations'][i-1].bbox_x1
        y1 = matfile['annotations'][i-1].bbox_y1
        x2 = matfile['annotations'][i-1].bbox_x2
        y2 = matfile['annotations'][i-1].bbox_y2
        img = cv2.imread(filename,cv2.IMREAD_COLOR)
        img = img[y1:y2, x1:x2]
        cv2.imwrite(filename, img)
    
    
if __name__  == '__main__':
    main()
    
