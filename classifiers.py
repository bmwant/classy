import os
import webbrowser

from collections import defaultdict 

import cv2
import numpy as np

from img_comp import ImageComparer
from classy import env, get_list_of_files, get_folders_in
from classy import TEST_FOLDER, INIT_FOLDER


def main():
    template = env.get_template('index.html')
    rt = classifiers()
    with open('result_table.html', 'w') as f:
        f.write(template.render(result=rt))
    webbrowser.open('result_table.html')

    
def classifiers():
    rt = {}  # Resulting table
    methods = ['mse', 'ssim']
    
    for test_pic in get_list_of_files(TEST_FOLDER, '.jpg'):
        res_for_pic = float('inf')  # best result for the picture
        class_num = {}  # guessed number of class for each picture
        pic_name = os.path.basename(test_pic)  # picture name e.g. 15.jpg
        
        train_array_knn = []
        test_array_knn = []
        labels_knn = []
        test_img = cv2.imread(test_pic)
        test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_array_knn.append(test_img_gray)
        
        for test_dir in get_folders_in(INIT_FOLDER, full=True):
            class_value = defaultdict(float)  # method -> summarized value of all pics by this method
            # method -> best class for the picture by this method 
            for compare_pic in get_list_of_files(test_dir, '.jpg'):
                #print(test_pic, compare_pic)
                
                pic = cv2.imread(compare_pic)
                pic_gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                train_array_knn.append(pic_gray)
                labels_knn.append([int(os.path.basename(test_dir))])
                
                ic = ImageComparer(test_pic, compare_pic)
                for method in methods:
                    cur_value = getattr(ic, method)()
                    class_value[method] += cur_value
            for method, value in class_value.items():
                if not method in class_num:
                    class_num[method] = {
                        'value': float('inf'),
                        'guess_class': None
                    }
                if class_num[method]['value'] > value:
                    class_num[method]['value'] = value
                    class_num[method]['guess_class'] = os.path.basename(test_dir)
        
        train_data_knn = np.array(train_array_knn)
        test_data_knn = np.array(test_array_knn)
        train_labels_knn = np.array(labels_knn)

        #70x70 = 4900 -> flatten picture to one-dimension array
        #you can also use flatten from neural_train.py
        train = train_data_knn.reshape(-1, 4900).astype(np.float32)  # train data for knn
        test = test_data_knn.reshape(-1, 4900).astype(np.float32)  # test data for use in knn

        knn = cv2.KNearest()
        knn.train(train, train_labels_knn)

        # distance, result, neighbours
        ret, result, neighbours, dist = knn.find_nearest(test, k=4)
        class_num['knn'] = {
            'value': str(dist[0]),
            'guess_class': int(result[0])
        }
        yield {pic_name: class_num}
        #rt[pic_name] = class_num

    
if __name__ == '__main__':
    main()