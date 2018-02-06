import os,time,cv2, sys, math
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
from scipy.misc import imread
import ast
#import utils

#labels_dir = '/media/lx/项目/DL_DATA/mass_buildings/train/map'
labels_dir = '/home/lx/Desktop/Semantic-Segmentation-Suite/CamVid/train_labels'
num_classes=12

#Initialize all the labels key with a list value
image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

label_to_frequency_dict = {}
for i in range(num_classes):
    label_to_frequency_dict[i] = []

print len(image_files)
for n in range(len(image_files)):
    image = imread(image_files[n])[:,:]
    unique_labels = list(np.unique(image))
    # print unique_labels
    #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
    for i in unique_labels:
        class_mask = np.equal(image, i)
        class_mask = class_mask.astype(np.float32)
        class_frequency = np.sum(class_mask)

        if class_frequency != 0.0:
            index = unique_labels.index(i)
            label_to_frequency_dict[index].append(class_frequency)

class_weights = []
# print(label_to_frequency_dict)

#Get the total pixels to calculate total_frequency later
total_pixels = 0
for frequencies in label_to_frequency_dict.values():
    total_pixels += sum(frequencies)

for i, j in label_to_frequency_dict.items():
    j = sorted(j) #To obtain the median, we've got to sort the frequencies

    median_frequency = np.median(j) / sum(j)
    total_frequency = sum(j) / total_pixels
    median_frequency_balanced = median_frequency / total_frequency
    class_weights.append(median_frequency_balanced)
class_weights /= np.min(class_weights)

print class_weights
