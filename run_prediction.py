#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from os.path import basename
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2 as cv
import tensorflow as tf
import argparse
import os

from helper2 import *
from utils2 import *

sys.path.append("models")
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from Encoder_Decoder_Skip import build_encoder_decoder_skip
from RefineNet import build_refinenet
from HF_FCN import build_hf_fcn

def get_predict(ortho, sess, num,
                l_ch, l_height, l_width,
                d_ch, d_height, d_width, offset=0):
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
        (n_patches, l_height, l_width, 1), dtype=np.float32)
    for i in range(n_patches):
        input_image = np.expand_dims(o_patches[i], axis=0)
        prob_image = sess.run(prob,feed_dict={input:input_image})
        pred_patches[i] = np.array(prob_image[0,:,:,1:])
        #pred class label images
        #output_image = np.array(prob_image[0,:,:,:])
        #print 'output_image.shape',output_image.shape
        #output_image = reverse_one_hot(output_image)
        #output_image = colour_code_segmentation(output_image)

    pred_img = np.zeros((h_limit, w_limit, 1), dtype=np.float32)
    for i, (rect, predict) in enumerate(zip(rects, pred_patches)):
        # predict = predict.swapaxes(0, 2).swapaxes(0, 1)
        pred_img[rect[0] + d_height / 2 - l_height / 2:
                 rect[0] + d_height / 2 + l_height / 2,
                 rect[1] + d_width / 2 - l_width / 2:
                 rect[1] + d_width / 2 + l_width / 2, :] = predict
    out_h = pred_img.shape[0] - (d_height - l_height)
    out_w = pred_img.shape[1] - (d_width - l_width)
    pred_img = pred_img[d_height / 2 - l_height / 2:out_h,
                        d_width / 2 - l_width / 2:out_w, :]
    ortho_img = ortho[d_height / 2 - l_height / 2 + offset:out_h,
                      d_width / 2 - l_width / 2 + offset:out_w, :]

    return pred_img, ortho_img

# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

if __name__ == '__main__':

    img_dir = './data/mass_buildings/test/sat'
    channel = 1
    device_id = 0
    offset = 0

    print("Start prediction ...")
    with tf.device('/gpu:1'):
        input = tf.placeholder(tf.float32,shape=[None,None,None,3])
        output = tf.placeholder(tf.float32,shape=[None,None,None,2])
        network = build_fc_densenet(input, preset_model = 'FC-DenseNet158', num_classes=2)
        prob = tf.nn.softmax(network)
    is_training = False
    continue_training = False
    class_names_string = "Building, Unlabelled"
    class_names_list = ["Building", "Unlabelled"]


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    saver=tf.train.Saver(max_to_keep=1000)
    # sess.run(tf.global_variables_initializer())

    count_params()

    if continue_training or not is_training:
        print('Loaded latest model checkpoint')
        saver.restore(sess, "checkpoints/latest_model.ckpt")

    result_dir = 'prediction_#9'
    print result_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    num = 1
    l_ch, l_height, l_width = channel, 512, 512
    d_ch, d_height, d_width = 3, 512, 512

    times = 0
    for img_fname in glob.glob('%s/*.tif*' % img_dir):
        ortho = cv.imread(img_fname,-1)
        ortho = ortho.astype('float32')
        ortho = ortho/255.0
        st = time.time()
        print 'origin ortho.shape',ortho.shape
        pred_img, ortho_img = get_predict(ortho, sess, num,
                                          l_ch, l_height, l_width,
                                          d_ch, d_height, d_width, offset)
        print time.time() - st, 'sec'
        times += time.time() - st
        print pred_img.shape
        pred_img_colour = pred_img>=0.5
        pred_img_colour = colour_code_segmentation(pred_img_colour)
        cv.imwrite('%s/pred_%d_%s.png' % (result_dir, offset, basename(img_fname)),pred_img * 125)
        cv.imwrite('%s/pred_colour_%d_%s.png' % (result_dir, offset, basename(img_fname)),pred_img_colour)
        cv.imwrite('%s/ortho_%d_%s.png' % (result_dir, offset, basename(img_fname)),ortho_img)
        np.save('%s/pred_%d_%s' % (result_dir, offset, basename(img_fname)),
                pred_img)
    
        print img_fname
    times /= 10
    print times
