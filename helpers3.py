from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools
import operator
import os

def get_class_list(list_path):
    """
    Retrieve the list of classes for the selected dataset.
    Note that the classes in the file must be LINE SEPARATED

    # Arguments
        list_path: The file path of the list of classes
        
    # Returns
        A python list of classes as strings
    """
    with open(list_path) as f:
        content = f.readlines()
    class_list = [x.strip() for x in content] 
    return class_list


def one_hot_it(label, num_classes=12):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        num_classes: The number of unique classes for this dataset
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    w = label.shape[0]
    h = label.shape[1]
    x = np.zeros([w,h,num_classes])
    unique_labels = np.unique(label)
    for i in range(0, w):
        for j in range(0, h):
            index = np.where(unique_labels==label[i][j])
            x[i,j,index]=1
    return x
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,1])

    for i in range(0, w):
        for j in range(0, h):
            index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
            x[i, j] = index
    return x



def colour_dict(x):
    """
    Dictionairy of colour codes for visualizing segmentation results

    # Arguments
        x: Value of the current pixel

    # Returns
        Colour code
    """
    return {
        0: [255, 255, 255],
        1: [0, 0, 255],
        2: [0, 255, 255],
        3: [0, 255, 0], 
        4: [255, 255, 0],
        5: [255, 0, 0],
        6: [0, 0, 0]
    }[x]

def colour_code_segmentation(image):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        
    # Returns
        Colour coded image for segmentation visualization
    """

    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,3])
    for i in range(0, w):
        for j in range(0, h):
            x[i, j, :] = colour_dict(image[i, j, 0])
    return x

