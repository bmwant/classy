import math
import operator

import cv2
import numpy as np
from skimage.measure import structural_similarity as ssim


class ImageComparer(object):
    def __init__(self, image1_input=None, image2_input=None):
        # Open image from path
        self.image1 = cv2.imread(image1_input)
        self.image2 = cv2.imread(image2_input)

    def mse(self):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        imageA = self.image1
        imageB = self.image2
        err = np.sum((imageA.astype('float') - imageB.astype('float')) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err
    
    def ssim(self):
        # the lower, the better
        image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        return 1.0 - ssim(image1, image2)
