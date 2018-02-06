#coding:utf-8
import sys
import shutil
import os
import glob
import numpy as np
import cv2 as cv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str)
args = parser.parse_args()
print args

def create_patches(sat_patch_size, map_patch_size, stride, map_ch,
                   sat_data_dir, map_data_dir,
                   sat_out_dir, map_out_dir, map_weihgt_dir):
    if os.path.exists(sat_out_dir):
        shutil.rmtree(sat_out_dir)
    if os.path.exists(map_out_dir):
        shutil.rmtree(map_out_dir)
    if os.path.exists(map_weihgt_dir):
        shutil.rmtree(map_weihgt_dir)
    os.makedirs(sat_out_dir)
    os.makedirs(map_out_dir)
    os.makedirs(map_weihgt_dir)

    # patch size
    sat_size = sat_patch_size
    map_size = map_patch_size
    print 'patch size:', sat_size, map_size, stride

    # get filenames
    sat_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % sat_data_dir)))
    map_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % map_data_dir)))
    index = np.arange(len(sat_fns))
    np.random.shuffle(index)
    sat_fns = sat_fns[index]
    map_fns = map_fns[index]

    n_all_files = len(sat_fns)
    print 'n_all_files:', n_all_files

    n_patches = 0
    for file_i, (sat_fn, map_fn) in enumerate(zip(sat_fns, map_fns)):
        if ((os.path.basename(sat_fn).split('.')[0])
                != (os.path.basename(map_fn).split('.')[0])):
            print 'File names are different',
            print sat_fn, map_fn
            return

        sat_im = cv.imread(sat_fn, cv.IMREAD_COLOR)
        map_im = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
        _, binary = cv.threshold(map_im, 50, 255, cv.THRESH_BINARY)
        canny = cv.Canny(binary, 60, 0, apertureSize = 3)
        _,contours,_ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        blank = np.ones((1500,1500))
        cv.drawContours(blank, contours, -1, 5, 1)
        pixel_weight = blank+2*(binary/255)
        # cv.imwrite('blank.png',blank*20)
        # cv.imwrite('binary.png',binary)
        # cv.imwrite('w.png',pixel_weight*20)
        for y in range(0, sat_im.shape[0] + stride, stride):
            for x in range(0, sat_im.shape[1] + stride, stride):
                if (y + sat_size) > sat_im.shape[0]:
                    y = sat_im.shape[0] - sat_size
                if (x + sat_size) > sat_im.shape[1]:
                    x = sat_im.shape[1] - sat_size

                sat_patch = np.copy(sat_im[y:y + sat_size, x:x + sat_size])
                map_patch = np.copy(map_im[y:y + sat_size, x:x + sat_size])
                pw_patch = np.copy(pixel_weight[y:y + sat_size, x:x + sat_size])
                # exclude patch including big white region
                if np.sum(np.sum(sat_patch, axis=2) == (255 * 3)) > 40:
                    continue

                #print sat_out_dir+os.path.basename(sat_fn).split('.')[0]+'_'+str(y)+'_'+str(x)+'.png', sat_patch.shape
                cv.imwrite(sat_out_dir+os.path.basename(sat_fn).split('.')[0]+'_'+str(y)+'_'+str(x)+'.png', sat_patch)
                cv.imwrite(map_out_dir+os.path.basename(map_fn).split('.')[0]+'_'+str(y)+'_'+str(x)+'.png', map_patch)
                cv.imwrite(map_weihgt_dir+os.path.basename(map_fn).split('.')[0]+'_'+str(y)+'_'+str(x)+'.png', pw_patch)
                n_patches += 1

        print file_i, '/', n_all_files, 'n_patches:', n_patches

    print 'patches:\t', n_patches


if __name__ == '__main__':
    create_patches(256, 256, 256, 1,
             args.dataset+'/mass_buildings/train/sat',
             args.dataset+'/mass_buildings/train/map',
             args.dataset+'/mass_buildings/patches256-2/train/',
             args.dataset+'/mass_buildings/patches256-2/train_labels/',
             args.dataset+'/mass_buildings/patches256-2/train_labels_weights/')
    # create_patches(256, 256, 64, 1,
    #                  args.dataset+'/mass_buildings/valid/sat',
    #                  args.dataset+'/mass_buildings/valid/map',
    #                  args.dataset+'/mass_buildings/patches256/val/',
    #                  args.dataset+'/mass_buildings/patches256/val_labels/')
    # create_patches(256, 256, 64, 1,
    #              args.dataset+'/mass_buildings/test/sat',
    #              args.dataset+'/mass_buildings/test/map',
    #              args.dataset+'/mass_buildings/patches256/test/',
    #              args.dataset+'/mass_buildings/patches256/test_labels/')
