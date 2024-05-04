import cv2
import math
import numpy as np

def guided_filter(im, p, r = 60, eps = 0.0001):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q


class DeweatherByDarkChannelPrior():
    def __init__(self):
        self.patch_size = 15

    def get_dark_channel(self, image):
        """
        """
        b,g,r = cv2.split(image)
        dc = cv2.min(cv2.min(r,g),b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(self.patch_size, self.patch_size))
        dark = cv2.erode(dc,kernel)
        return dark

    def get_atmospheric_light(self, image, dark_channel):
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

    def estimate_transmission(self, image, atm_light, omega):
        """
        """
        im3 = np.empty(image.shape,image.dtype)

        for ind in range(0,3):
            im3[:,:,ind] = image[:,:,ind]/atm_light[0,ind]

        transmission = 1 - omega*self.get_dark_channel(im3)
        return transmission

    def refine_transmission(self, image, transmission_estimate):
        """
        """
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)/255
        t = guided_filter(gray, transmission_estimate)
        return t

    def recover(self, im, t, A, tx = 0.1):
        """
        """
        res = np.empty(im.shape,im.dtype)
        t = cv2.max(t,tx)

        for ind in range(0,3):
            res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

        return res

    def deweather(self, image):
        """
        """
        image = image.astype('float64')/255
        dark_channel = self.get_dark_channel(image)
        atm_light = self.get_atmospheric_light(image, dark_channel)
        transmission_estimate = self.estimate_transmission(image, atm_light, omega=0.95)
        transmission = self.refine_transmission(src, transmission_estimate)
        J = self.recover(image, transmission, atm_light, 0.01)
        return J

if __name__ == '__main__':
    fn = '/shortdata/ziwang/projects/nerf-factory/data/carla/output_foggy_world_new/images/00000.png'

    src = cv2.imread(fn)

    deweather = DeweatherByDarkChannelPrior()
    J = deweather.deweather(src)


    # cv2.imshow("dark",dark)
    # cv2.imshow("t",t)
    # cv2.imshow('I',src)
    # cv2.imshow('J',J)
    cv2.imwrite("/shortdata/ziwang/projects/nerf-factory/figs/J.png",J*255)
    # cv2.waitKey()