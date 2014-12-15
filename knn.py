__author__ = 'Most Wanted'
import os
import cv2
import numpy as np
from classy import TEST_FOLDER, INIT_FOLDER
from classy import get_list_of_files


def knn():
    train_folder = INIT_FOLDER
    test_folder = TEST_FOLDER

    train_array = []
    labels = []
    for folder_name in os.listdir(train_folder):
        next_dir = os.path.join(train_folder, folder_name)
        if os.path.isdir(next_dir):
            for image in os.listdir(next_dir):
                if image.endswith('.jpg'):
                    image_file = os.path.join(next_dir, image)
                    img = cv2.imread(image_file)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    train_array.append(gray)
                    labels.append([int(folder_name)])

    test_array = []
    image_names = []
    for image in get_list_of_files(test_folder, '.jpg'):
        image_file = os.path.join(test_folder, image)
        image_names.append(image_file)
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_array.append(gray)

    train_data = np.array(train_array)
    test_data = np.array(test_array)
    #70x70 = 4900
    traind = train_data
    #traind, testd = np.vsplit(train_data, 2)

    #print('All data array size: ', train_data.shape)
    #print('Splitted data array size: ', traind.shape)

    
    new_train_labels = np.array(labels)

    #ntl, _ = np.vsplit(new_train_labels, 2)
    #print('New train labels array size: ', ntl.shape)
    #print('New train labels[0][0] ', type(ntl[0][0]))
    # Initiate kNN, train the data, then test it with test data for k=1
    train = train_data.reshape(-1, 4900).astype(np.float32)
    test = test_data.reshape(-1, 4900).astype(np.float32)
    
    knn = cv2.KNearest()
    knn.train(train, new_train_labels)

    # distance, result, neighbours

    ret, result, neighbours, dist = knn.find_nearest(test, k=4)
    print(result)
    """
    train = train_data.reshape(-1, 4900).astype(np.float32)
    test = test_data.reshape(-1, 4900).astype(np.float32)
    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row,100) for row in np.vsplit(gray, 50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Now we prepare train_data and test_data.
    train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
    test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k,250)[:,np.newaxis]
    test_labels = train_labels.copy()

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train,train_labels)
    ret,result,neighbours,dist = knn.find_nearest(test,k=5)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print accuracy
    """
    
if __name__ == '__main__':
    knn()