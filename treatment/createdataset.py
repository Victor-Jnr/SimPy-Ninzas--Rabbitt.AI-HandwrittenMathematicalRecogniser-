"""
Uses the directory for dots: main_dir_dot = treatment_dir + '/datasets/dots/'
Prints the debug message "Dot all" indicating that it is processing dot images.
In the dataset preparation function, every image is assigned the label "28". Optional label "29"
Saves the files with names starting with dot_, such as "dot_training_images.npz", "dot_testing_labels.npz", etc.

Summary:Victor-Jnr
"""


from sklearn.utils import shuffle
from mathreader.image_processing import preprocessing as preprocessing
import numpy as np
import cv2
import imutils
import os
import re
import random
import sys
import math
import json
import idx2numpy
from threading import Thread

treatment_dir = 'treatment/'
treated_dir = 'treated_data/'

if not os.path.exists(treatment_dir + treated_dir):
    os.mkdir(treatment_dir + treated_dir)

main_dir_dot = treatment_dir + '/datasets/dots/'
main_dir = 'treatment/datasets/handwrittenmathsymbols_all/'

configs_dot = {
    'black': False,
    'dilate': True,
    'dataset': False,
    'resize': 'smaller'
}

configs = {
    'black': False,
    'dilate': True,
    'dataset': True,
    'resize': 'smaller'
}

dirs = ['0/', '1/', '2/', '3/', '4/', '5/', '6/', '7/', '8/', '9/', '-/', '(/', ')/', '[/', ']/', '{/', '}/', '+/', 'a/', 'b/', 'c/', 'm/', 'n/', 'sqrt/', 'x/', 'y/', 'z/', 'neq/']

labels = {
    '0/': '0', '1/': '1', '2/': '2', '3/': '3', '4/': '4', '5/': '5', '6/': '6', '7/': '7', '8/': '8', '9/': '9',
    "-/": "10", "(/": "11", ")/": "12", "[/": "13", "]/": "14", "{/": "15", "}/": "16", "+/": "17", "a/": "18",
    "b/": "19", "c/": "20", "m/": "21", "n/": "22", "sqrt/": "23", "x/": "24", "y/": "25", "z/": "26", "neq/": "27"
}

print("Dot all")

def get_labels():
    try:
        with open('docs/config/config_all.json') as json_file:
            labels_json = json_file.read()
            labels_dict = json.loads(labels_json)
            labels = labels_dict
    except Exception as e:
        print(e)
        labels = {}

    return labels

def save_json_file(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data))

def dots():
    images = []
    for filename in os.listdir(main_dir_dot):
        print('filename: %s' % filename)
        if re.search(r"\.(jpg|jpeg|png)$", filename, re.IGNORECASE):
            objs = preprocessing.ImagePreprocessing(configs_dot).treatment(main_dir_dot + filename)
            for obj in objs[0]:
                try:
                    images.append(obj['image'])
                except BaseException:
                    pass
    return images

def prepara_dataset(images):
    labels = ['28' for _ in images]  

    images = shuffle(images)
    amount = len(images)

    training_size = math.floor(amount * 0.7)

    training_images = np.asarray(images[:training_size])
    training_labels = np.asarray(labels[:training_size])

    testing_images = np.asarray(images[training_size:])
    testing_labels = np.asarray(labels[training_size:])

    print('Training:')
    print(training_images.shape)
    print('Test:')
    print(testing_images.shape)

    np.savez(treatment_dir + treated_dir + "dot_training_images", training_images)
    np.savez(treatment_dir + treated_dir + "dot_training_labels", training_labels)
    np.savez(treatment_dir + treated_dir + "dot_testing_images", testing_images)
    np.savez(treatment_dir + treated_dir + "dot_testing_labels", testing_labels)

images = dots()
print('Total:')
print(len(images))
prepara_dataset(images)

print('Kaggle all')

def parallel(interval, tid, training_images, training_labels, testing_images, testing_labels):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for j in range(interval[0], interval[1]+1):
        folder = dirs[j]
        file_list = os.listdir(main_dir + folder)
        count = 1
        amount = len(file_list)
        training_size = math.floor(amount * 0.8)
        testing_size = math.floor(amount * 0.2)

        for filename in file_list:
            print('filename: %s' % filename)
            if re.search(r"\.(jpg|jpeg|png)$", filename, re.IGNORECASE):
                image_path = main_dir + folder + filename
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                image = preprocessing.ImagePreprocessing(configs).treatment_without_segmentation(image)
                if len(image) == 0:
                    print("EMPTY: ", image_path)
                    continue

                if count <= training_size:
                    train_images.append(image)
                    train_labels.append(labels[folder])
                elif count <= training_size + testing_size:
                    test_images.append(image)
                    test_labels.append(labels[folder])
                else:
                    break

                count += 1

    training_images.extend(train_images)
    training_labels.extend(train_labels)
    testing_images.extend(test_images)
    testing_labels.extend(test_labels)

def get_symbols():
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []

    size = len(dirs) // 16 or 1
    remain = len(dirs) % 16
    initial = 0
    threads = []

    for i in range(size, len(dirs) + 1, size):
        if i == len(dirs) - remain:
            interval = (initial, i - 1 + remain)
        else:
            interval = (initial, i - 1)

        t = Thread(target=parallel, args=(interval, i, training_images, training_labels, testing_images, testing_labels))
        threads.append(t)
        initial += size

    for t in threads:
        print('Starting Threads')
        t.start()

    for t in threads:
        print('Processing threads...')
        t.join()

    print('All Threads Completed')

    training_labels, training_images = shuffle(training_labels, training_images)
    testing_labels, testing_images = shuffle(testing_labels, testing_images)

    training_images = np.asarray(training_images)
    training_labels = np.asarray(training_labels)
    testing_images = np.asarray(testing_images)
    testing_labels = np.asarray(testing_labels)

    print('Training:')
    print(training_images.shape)
    print('Test:')
    print(testing_images.shape)

    np.savez(treatment_dir + treated_dir + "kaggle_all_training_images", training_images)
    np.savez(treatment_dir + treated_dir + "kaggle_all_training_labels", training_labels)
    np.savez(treatment_dir + treated_dir + "kaggle_all_testing_images", testing_images)
    np.savez(treatment_dir + treated_dir + "kaggle_all_testing_labels", testing_labels)

get_symbols()
