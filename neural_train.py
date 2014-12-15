import os

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

import cv2

from classy import env, get_list_of_files, get_folders_in
from classy import TEST_FOLDER, INIT_FOLDER


RESIZE = 10
net = None

def load_image_arr(path):
    img = cv2.imread(path)
    height, width = img.shape[:2]
    res = cv2.resize(img,(width/RESIZE, height/RESIZE), interpolation=cv2.INTER_CUBIC)
    return flatten(res)


def flatten(img):
    result = []
    for el in img:
        # isinstance collections.Iterable +
        if hasattr(el, '__iter__') and not isinstance(el, basestring):
            result.extend(flatten(el))
        else: result.append(el)
    return result


def train_net():
    t = load_image_arr('example.jpg')  # example of image used by network
    #print('Resized t length:', len(t))
    global net
    net = buildNetwork(len(t), len(t), 1)
    ds = SupervisedDataSet(len(t), 1)
    
    for test_dir in get_folders_in(INIT_FOLDER, full=True):
        img_class = os.path.basename(test_dir)
        for test_pic in get_list_of_files(test_dir, '.jpg'):
            #print('Adding {0} with class {1}'.format(test_pic, img_class))
            ds.addSample(load_image_arr(test_pic), (img_class, ))  # <- class
    
    trainer = BackpropTrainer(net, ds)
    error = 10
    iteration = 0
    while error > 0.1:
        error = trainer.train()
        iteration += 1
        yield 'Iteration: {0}. Error: {1}'.format(iteration, error)
        
        
def activate_net():
    global net
    for test_pic in get_list_of_files(TEST_FOLDER, '.jpg'):
        pic_name = os.path.basename(test_pic)
        result = net.activate(load_image_arr(test_pic))[0]
        yield pic_name, result

    
if __name__ == '__main__':
    list(train_net())
    for mes in activate_net():
        print(mes)
    