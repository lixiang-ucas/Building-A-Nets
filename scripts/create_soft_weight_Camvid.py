#coding:utf-8
import sys
import shutil
import os
import glob
import numpy as np
import cv2 as cv
import utils


map_data_dir = 'CamVid/train_labels/'
map_weihgt_dir = 'CamVid/train_labels_weights/'
if os.path.exists(map_weihgt_dir):
    shutil.rmtree(map_weihgt_dir)
os.makedirs(map_weihgt_dir)

# get filenames
map_fns = np.asarray(sorted(glob.glob('%s/*.png*' % map_data_dir)))

n_all_files = len(map_fns)
print 'n_all_files:', n_all_files

balance_weight,cls_freq = utils.median_frequency_balancing('CamVid/train_labels/')

for map_fn in map_fns:
    print map_weihgt_dir+os.path.basename(map_fn).split('.')[0]+'.png'
    map_im = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
    # _, binary = cv.threshold(11-map_im, 0, 255, cv.THRESH_BINARY)
    # canny = cv.Canny(binary, 60, 0, apertureSize = 3)
    # _,contours,_ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # blank = np.ones(map_im.shape)
    # cv.drawContours(blank, contours, -1, 5, 1)
    # pixel_weight = blank+2*(binary/255)
    pixel_weight = np.empty(map_im.shape, dtype='float32')
    for c in range(len(cls_freq)):
        if c!=12:
            pixel_weight[map_im==c]=max(2,1/cls_freq[c])
        else:
            pixel_weight[map_im==c]=1
    cv.imwrite(map_weihgt_dir+os.path.basename(map_fn).split('.')[0]+'.png', pixel_weight)