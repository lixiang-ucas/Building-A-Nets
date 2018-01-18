#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, time
from os.path import basename
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2 as cv
import tensorflow as tf
import argparse
import os

sys.path.append("models")
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from Encoder_Decoder_Skip import build_encoder_decoder_skip
from RefineNet import build_refinenet
from HF_FCN import build_hf_fcn


def colour_dict(x):
    """
    Dictionairy of colour codes for visualizing segmentation results

    # Arguments
        x: Value of the current pixel

    # Returns
        Colour code
    """

    # Color palette
    palette = {0: (255, 255, 255),  # Impervious surfaces (white)
           1: (0, 0, 255),      # Buildings (dark blue)
           2: (0, 255, 255),    # Low vegetation (light blue)
           3: (0, 255, 0),      # Tree (green)
           4: (255, 255, 0),    # Car (yellow)
           5: (255, 0, 0),      # Clutter (red)
           6: (0, 0, 0)}  

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
            x[i, j, :] = colour_dict(image[i, j])
    return x

# process prediction on full test images
def get_predict(ortho, sess, num_classes, l_ch, l_height, l_width, d_ch, d_height, d_width, offset=0):
    h_limit = ortho.shape[0]
    w_limit = ortho.shape[1]

    # create input, label patches
    rects = []  # input data region
    o_patches = []
    for y in range(offset, h_limit, l_height):
        
        for x in range(offset, w_limit, l_width):
            if (y + d_height > h_limit):
                y = h_limit - d_height
            if (x + d_width > w_limit):
                x = w_limit - d_width
            rects.append((y - offset, x - offset,
                          y - offset + d_height, x - offset + d_width))
            # ortho patch
            o_patch = ortho[y:y + d_height, x:x + d_width, :]
            # o_patch = o_patch.swapaxes(0, 2).swapaxes(1, 2)
            o_patches.append(o_patch)

    o_patches = np.asarray(o_patches, dtype=np.float32)

    # the number of patches
    n_patches = len(o_patches)
    print 'n_patches %s' % n_patches

    # create predict, label patches
    pred_patches = np.zeros(
        (n_patches, l_height, l_width, num_classes), dtype=np.float32)
    for i in range(n_patches):
        input_image = np.expand_dims(o_patches[i], axis=0)
        prob_image = sess.run(prob,feed_dict={input:input_image})
        pred_patches[i] = np.array(prob_image[0])

    pred_img = np.zeros((h_limit, w_limit, num_classes), dtype=np.float32)
    for i, (rect, predict) in enumerate(zip(rects, pred_patches)):
        pred_img[rect[0] + d_height / 2 - l_height / 2:
             rect[0] + d_height / 2 + l_height / 2,
             rect[1] + d_width / 2 - l_width / 2:
             rect[1] + d_width / 2 + l_width / 2, :] += predict
    out_h = pred_img.shape[0] - (d_height - l_height)
    out_w = pred_img.shape[1] - (d_width - l_width)
    pred_img = pred_img[d_height / 2 - l_height / 2:out_h,
                        d_width / 2 - l_width / 2:out_w, :]
    ortho_img = ortho[d_height / 2 - l_height / 2 + offset:out_h,
                      d_width / 2 - l_width / 2 + offset:out_w, :]

    return pred_img, ortho_img

def metrics(predictions, gts):
    """ Compute the metrics from the RGB-encoded predictions and ground truthes
    Args:
        predictions (array list): list of RGB-encoded predictions (2D maps)
        gts (array list): list of RGB-encoded ground truthes (2D maps, same dims)
    """
    prediction_labels = np.concatenate([l.flatten() for l in predictions])
    gt_labels = np.concatenate([l.flatten() for l in gts])

    cm = confusion_matrix(
            gt_labels,
            prediction_labels,
            range(len(label_values)))

    print "Confusion matrix :"
    print cm
    print "---"
    # Compute global accuracy
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    total = sum(sum(cm))
    print "{} pixels processed".format(total)
    print "Total accuracy : {}%".format(accuracy * 100 / float(total))
    print "---"
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in xrange(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print "F1Score :"
    for l_id, score in enumerate(F1Score):
        print "{}: {}".format(label_values[l_id], score)

    print "---"
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print "Kappa: " + str(kappa)

if __name__ == '__main__':
    gpu_id = 0
    num_classes = 6
    infer_ids = [32,34,37]
    print("Start prediction ...")
    with tf.device('/gpu:'+str(gpu_id)):
        input = tf.placeholder(tf.float32,shape=[None,None,None,3])
        output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
        network = build_fc_densenet(input, preset_model = 'FC-DenseNet103', num_classes=num_classes)
        prob = tf.nn.softmax(network)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    saver=tf.train.Saver(max_to_keep=1000)
    # sess.run(tf.global_variables_initializer())

    print('Loaded latest model checkpoint')
    saver.restore(sess, "checkpoints6-2/latest_model.ckpt")

    result_dir = 'prediction_#6'
    print result_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    offset = 0
    l_ch, l_height, l_width = 1, 512, 512
    d_ch, d_height, d_width = 3, 512, 512

    times = 0
    pred_label_list = []
    gt_label_list = []
    BASE_FOLDER = '/media/zhoun/Data/lx/caffe/DeepNetsForEO/ISPRS/Vaihingen/'
    print "Processing {} images...".format(len(infer_ids),-1)
    for l in infer_ids:
        img_fname = 'top_mosaic_09cm_area{}.tif'.format(l)
        ortho = cv.imread(BASE_FOLDER + 'top/' + img_fname,-1)
        ortho = ortho.astype('float32')
        ortho = ortho/255.0
        gt_label = cv.imread(BASE_FOLDER + 'gts_numpy/top_mosaic_09cm_area{}.tif'.format(l),-1)
        st = time.time()
        print 'origin ortho.shape',ortho.shape
        pred_img, ortho_img = get_predict(ortho, sess, num_classes,
                                          l_ch, l_height, l_width,
                                          d_ch, d_height, d_width, offset)
        print time.time() - st, 'sec'
        times += time.time() - st
        #pred class label images
        pred_label = np.argmax(pred_img,axis=2)
        pred_img_colour = colour_code_segmentation(pred_label)
        # cv.imwrite('%s/pred_%d_%s.png' % (result_dir, offset, basename(img_fname)),pred_img * 125)
        cv.imwrite('%s/pred_colour_%d_%s.png' % (result_dir, offset, basename(img_fname)),pred_img_colour)
        cv.imwrite('%s/ortho_%d_%s.png' % (result_dir, offset, basename(img_fname)),ortho_img)
        np.save('%s/pred_%d_%s' % (result_dir, offset, basename(img_fname)),
                pred_img)
    
        print img_fname

        pred_label_list.append(pred_label)
        gt_label_list.append(gt_label)
    times /= 10
    print times
    print "Computing metrics..."
    metrics(pred_label_list, gt_label_list)