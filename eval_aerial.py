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
from sklearn.metrics import confusion_matrix
import utils

label_values = ['Unlabeled', 'building']

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

    result_dir =  'prediction_valid_#107-0089'

    pred_label_list = []
    gt_label_list = []

    BASE_FOLDER = './data/AerialImageDataset/valid/gt/'

#     for img_fname in glob.glob('%s/pred_0_*.tif.npy' % result_dir):
#         pred_label = np.load(img_fname)
#         pred_label = pred_label>=0.5

#         img_fname = os.path.basename(img_fname)[7:].split('.npy')[0]
#         print(img_fname)
#         gt_label = cv.imread(BASE_FOLDER + img_fname, cv.IMREAD_GRAYSCALE)
#         gt_label /= gt_label.max()
#         print(np.unique(pred_label),np.unique(gt_label))
#         pred_label_list.append(pred_label)
#         gt_label_list.append(gt_label)

    for img_fname in glob.glob('%s/*.tif' % BASE_FOLDER):
        gt_label = cv.imread(img_fname, cv.IMREAD_GRAYSCALE)
        gt_label /= gt_label.max()
        img_fname = os.path.basename(img_fname)
        pred_label = np.load(result_dir + '/pred_0_'+img_fname+'.npy')
        pred_label = pred_label>=0.5
        print(np.unique(pred_label),np.unique(gt_label))
        pred_label_list.append(pred_label)
        gt_label_list.append(gt_label)

    print "Computing metrics..."
    # metrics(pred_label_list, gt_label_list)
    pred = np.concatenate([l.flatten() for l in pred_label_list])
    label = np.concatenate([l.flatten() for l in gt_label_list])

    TP = np.float(np.count_nonzero(pred * label)) + 1.0
    TN = np.float(np.count_nonzero((pred-1) * (label-1)))
    FP = np.float(np.count_nonzero(pred * (label - 1))) + 1.0
    prec = TP / (TP + FP)
    FN = np.float(np.count_nonzero((pred - 1) * label)) + 1.0
    rec = TP / (TP + FN)
    iou = TP / (TP+FN+FP)
    f1 = np.divide(2 * prec * rec, (prec + rec))
    acc = (TP+TN)/(TP+TN+FP+FN)
    # print(prediction_labels.shape,gt_labels.shape)
    # acc = utils.compute_avg_accuracy(prediction_labels,gt_labels)
    # rec = utils.recall(prediction_labels,gt_labels)
    # prec = utils.precision(prediction_labels,gt_labels)
    # iou = utils.compute_mean_iou(prediction_labels,gt_labels)
    print('acc,rec,prec,f1,iou:',acc,prec,rec,f1,iou)

