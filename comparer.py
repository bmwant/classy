import os
import math
import operator
import hashlib
import pprint
import pickle
import shutil
import logging
import copy

from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


from PIL import Image, ImageChops
from grabber import Grabber


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


formatter = logging.Formatter('%(name)s:%(asctime)s:[%(levelname)s] %(message)s')
formatter.datefmt = '%d/%m/%y %H:%M'
handler = logging.StreamHandler()
handler.setFormatter(formatter)
log.addHandler(handler)

class ImageComparer(object):
    def __init__(self, image1_input=None, image2_input=None):
        # Not safe but for development
        if isinstance(image1_input, str):
            self.image1 = Image.open(image1_input)
            self.image2 = Image.open(image2_input)
        else:
            self.image1 = image1_input
            self.image2 = image2_input
    
    def compare(self):
        """
        print('Using dhash(): ')
        print(self.dhash(self.image1) == self.dhash(self.image2))
        
        print('Using histogram distance: ')
        """
        h1 = self.image1.histogram()
        h2 = self.image2.histogram()

        "Calculate the root-mean-square difference between two images"
        rms = math.sqrt(reduce(operator.add,
            map(lambda a,b: (a-b)**2, h1, h2))/len(h1))
        return rms
        
        """
        print('Exact comparison: ')
        print(self.equal(self.image1, self.image2))
        """
        

    def compare_for_cv(self, image_a, image_b):
        rms = math.sqrt(reduce(operator.add,
            map(lambda a,b: (a-b)**2, image_a, image_b))/len(image_a))
        return rms

    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def comp_qt(self, image_a, image_b):
        m = self.mse(image_a, image_b)
        s = ssim(image_a, image_b)
        return m

    def compare_images(self, imageA, imageB, title):
        # compute the mean squared error and structural similarity
        # index for the images
        m = self.mse(imageA, imageB)
        s = ssim(imageA, imageB)

        # setup the figure
        fig = plt.figure(title)
        plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap=plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap=plt.cm.gray)
        plt.axis("off")

        # show the images
        plt.show()

    @staticmethod
    def equal(im1, im2):
        return ImageChops.difference(im1, im2).getbbox() is None
    
    @staticmethod
    def resize_to_grayscale(image, size=8):
        result = image.convert('L').resize(
            (size + 1, size),
            Image.ANTIALIAS,
        )
        return result
    
    @staticmethod
    def dhash(image, hash_size=8):
        # Grayscale and shrink the image in one step.
        image = image.convert('L').resize(
            (hash_size + 1, hash_size),
            Image.ANTIALIAS,
        )
    
        pixels = list(image.getdata())
    
        # Compare adjacent pixels.
        difference = []
        for row in xrange(hash_size):
            for col in xrange(hash_size):
                pixel_left = image.getpixel((col, row))
                pixel_right = image.getpixel((col + 1, row))
                difference.append(pixel_left > pixel_right)
    
        # Convert the binary array to a hexadecimal string.
        decimal_value = 0
        hex_string = []
        for index, value in enumerate(difference):
            if value:
                decimal_value += 2**(index % 8)
            if (index % 8) == 7:
                hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
                decimal_value = 0
    
        return ''.join(hex_string)
    




    
def test_all_rms():
    #img = Image.open(r'C:\tempme\data\6cfabbc7b2a1f6fc9734b0cc2fffdb30.jpg')
    #c = ImageComparer.resize_to_grayscale(img, size=20)
    #c.save(r'C:\tempme\res.jpg')

    for f1 in Grabber.get_list_of_files(r'C:\tempme\data', '.jpg'):
        #print(f1, '----------')
        lst = []
        for f2 in Grabber.get_list_of_files(r'C:\tempme\grabbed_once', '.jpg'):
            ff1 = Image.open(f1)
            ff2 = Image.open(f2)
            c1 = ImageComparer.resize_to_grayscale(ff1, size=30)
            c2 = ImageComparer.resize_to_grayscale(ff2, size=30)
            icmp = ImageComparer(c1, c2)
            lst.append(icmp.compare())
        k = sorted(lst)
        print(k)

def get_list_of_files(directory, ext):
    files = []
    for file in os.listdir(directory):
        if file.endswith(ext):
            files.append(os.path.join(directory, file))
    return files


def normalize(table):
    for key, value in table.iteritems():
        new_value = sum(value) / len(value)
        table[key] = new_value
    return table


def locate_probable(table, where, filename):
    probable_folder = min(table.iterkeys(), key=lambda k: table[k])
    coef = table[probable_folder]
    log.info('Probable folder %s with coef: %s' % (probable_folder, coef))
    result_folder = os.path.join(where, probable_folder)
    # create new name for file based on hash
    with open(filename, 'rb') as f:
        image_file = f.read()
        img_hash = hashlib.md5(image_file).hexdigest()

    if coef > 30:
        nfl = r'D:\coding\senderbot\new_unique'
        log.info('Get new image!')
        shutil.move(filename, nfl)
    else:
        shutil.copy(filename, os.path.join(result_folder, '{0}.jpg'.format(img_hash)))


def get_next_folder():
    base_directory = os.path.join(os.path.dirname(__file__), 'base')
    max_folder_number = 0
    for folder_name in os.listdir(base_directory):
        next_dir = os.path.join(base_directory, folder_name)
        if os.path.isdir(next_dir):
            next_number = int(folder_name)
            if next_number > max_folder_number:
                max_folder_number = next_number
    return os.path.join(base_directory, str(max_folder_number+1))

def compare_to_base(filename):
    base_directory = os.path.join(os.path.dirname(__file__), 'base')
    table_cmp = {}
    for foldername in os.listdir(base_directory):
        next_dir = os.path.join(base_directory, foldername)
        if os.path.isdir(next_dir):
            for image in os.listdir(next_dir):
                if image.endswith('.jpg'):
                    image_path = os.path.join(next_dir, image)
                    icmp = ImageComparer(filename, image_path)
                    rms = icmp.compare()
                    if foldername in table_cmp:
                        table_cmp[foldername].append(rms)
                    else:
                        table_cmp[foldername] = [rms]

    #table = normalize(table_cmp)
    return table_cmp


def collect_base():
    tmp_folder = r'D:\coding\senderbot\grabbed_once'
    for img in Grabber.get_list_of_files(tmp_folder, '.jpg'):
        base_directory = os.path.join(os.path.dirname(__file__), 'base')
        table = compare_to_base(img)
        img_file = os.path.join(tmp_folder, img)
        locate_probable(table, base_directory, img_file)


def try_to_guess():
    tmp_folder = r'D:\coding\senderbot\try'
    counter = 0
    newc = 0
    for img in Grabber.get_list_of_files(tmp_folder, '.jpg'):
        img_file = os.path.join(tmp_folder, img)
        print 'Analyzing file: %s' % img
        tbl = compare_to_base(img_file)
        tbl2 = copy.deepcopy(tbl)
        tbl2 = normalize(tbl2)
        min_coef = 100
        min_avg = 100
        index = None
        for key in tbl:
            min_tmp = min(tbl[key])
            min_avg_tmp = tbl2[key]
            if min_tmp < min_coef and min_avg_tmp < min_avg:
                index = key
                min_coef = min_tmp
                min_avg = min_avg_tmp

        with open(img_file, 'rb') as f:
            image_file = f.read()
            img_hash = hashlib.md5(image_file).hexdigest()


        nfl = '{0}.jpg'.format(img_hash)  # new file name
        if min_coef < 15 and min_avg < 30:
            log.info('Probably: %s with min: %s and average: %s' % (index, min_coef, min_avg))
            base_directory = os.path.join(os.path.dirname(__file__), 'base')
            result_folder = os.path.join(base_directory, index)
            shutil.move(img_file, os.path.join(result_folder, nfl))
            counter += 1
        else:
            log.info('Maybe new, because: %s and avg: %s' % (min_coef, min_avg))
            folder_for_new_unique = r'D:\coding\senderbot\new_unique'
            shutil.move(img_file, os.path.join(folder_for_new_unique, nfl))
            newc += 1

    log.info('added already known pictures to base: %s' % counter)
    log.info('unique pictures here: %s' % newc)

def main():
    #icmp = ImageComparer(r'C:\tempme\data\6cfabbc7b2a1f6fc9734b0cc2fffdb30.jpg', r'C:\tempme\grabbed_once\9_new.jpg')
    #test_all_rms()
    #collect_base()
    #try_to_guess()


    one = cv2.imread("one.jpg")
    two = cv2.imread("two.jpg")
    three = cv2.imread("three.jpg")

    # convert the images to grayscale
    one = cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)
    two = cv2.cvtColor(two, cv2.COLOR_BGR2GRAY)
    three = cv2.cvtColor(three, cv2.COLOR_BGR2GRAY)

    icmp = ImageComparer(one, two)

    print(icmp.compare_for_cv(one, two))

    # initialize the figure
    fig = plt.figure("Images")
    images = ("Original", one), ("Two", two), ("Three", three)

    # loop over the images
    for (i, (name, image)) in enumerate(images):
        # show the image
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis("off")

    # show the figure
    plt.show()

    # compare the images
    icmp.compare_images(one, one, "Original vs. Original")
    icmp.compare_images(one, two, "Original vs. Two")
    icmp.compare_images(one, three, "Original vs. Three")

    
    
if __name__ == '__main__':
    main()