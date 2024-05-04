import numpy as np  
import cv2
import os
import math

def get_dark_channel(image):
    """
    """
    b,g,r = cv2.split(image)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))
    dark = cv2.erode(dc,kernel)
    return dark
    
def get_atmospheric_light(image, dark_channel):
    """
    """
    [h,w] = image.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark_channel.reshape(imsz)
    imvec = image.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

if __name__ == '__main__':
    img_path = '/shortdata/ziwang/projects/nerf-factory/data/carla/output_clear_world_with_depth_1822-1999_physics_fog/images/000000.png'
    img = cv2.imread(img_path)
    dark = get_dark_channel(img)
    A = get_atmospheric_light(img, dark)
    
    print(A)