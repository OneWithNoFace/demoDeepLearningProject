import os
import cv2

def read_images(folder_path):
  for fname in os.listdir(folder_path):
    img1 = cv2.imread(os.path.join(folder_path,fname))
    input_shape = (256,256,3)
    img2 = cv2.copyMakeBorder(img1,0,input_shape[0]-img1.shape[0],0,input_shape[1]-img1.shape[1],cv2.BORDER_CONSTANT)
    print(img1.shape,img2.shape)
    cv2.imwrite(os.path.join(folder_path,fname),img1 if input_shape  == img1.shape else img2)

read_images('../data/train/')
read_images('../data/test/')