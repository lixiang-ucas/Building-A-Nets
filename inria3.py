#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys

import helpers 
import utils 
from collections import deque
from models import *

import matplotlib.pyplot as plt
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

sys.path.append("models")
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from RefineNet import build_refinenet
from FRRN import build_frrn
from MobileUNet import build_mobile_unet
from PSPNet import build_pspnet
from GCN import build_gcn
from HF_FCN import build_hf_fcn


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--is_training', type=str2bool, default=True, help='Whether we are training or testing')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="./data/AerialImageDataset/patches256", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=256, help='Height of input image to network')
parser.add_argument('--crop_width', type=int, default=256, help='Width of input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Width of input image to network')
parser.add_argument('--num_val_images', type=int, default=500, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=None, help='Whether to randomly zoom in for data augmentation')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. Currently supports: FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, FC-DenseNet158, FC-DenseNet232, HF-FCN, Encoder-Decoder, Encoder-Decoder-Skip, RefineNet-Res50, RefineNet-Res101, RefineNet-Res152, FRRN-A, FRRN-B, MobileUNet, MobileUNet-Skip, PSPNet-Res50, PSPNet-Res101, PSPNet-Res152, GCN-Res50, GCN-Res101, GCN-Res152, custom')
parser.add_argument('--exp_id', type=int, default=1, help='Number of experiments')
parser.add_argument('--gpu_ids', type=int, default=0, help='Set GPU device id')
parser.add_argument('--is_BC', type=str2bool, default=True, help='whegher to use balanced weight')
parser.add_argument('--is_balanced_weight', type=str2bool, default=False, help='whegher to use balanced weight')
parser.add_argument('--is_edge_weight', type=str2bool, default=False, help='whegher to use balanced weight')

args = parser.parse_args()


# Get a list of the training, validation, and testing file paths
def prepare_data(dataset_dir=args.dataset):
    train_input_names=[]
    train_output_names=[]
    train_output_weight_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names

def create_record(dataset_dir=args.dataset):
    train_input_names=[]
    train_output_names=[]
    train_output_weight_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)

    writer = tf.python_io.TFRecordWriter("train.tfrecords")

    for t_ind in range(len(train_input_names)):
        img = cv2.imread(train_input_names[t_ind],-1)
        label = cv2.imread(train_output_names[t_ind],-1)
        label = helpers.one_hot_it(label=label, num_classes=2).astype('uint8')
        # print(img.shape,img.dtype,label.shape,label.dtype)
        img_raw = img.tobytes() #将图片转化为原生bytes
        label_raw = label.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            "label_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    
    writer = tf.python_io.TFRecordWriter("valid.tfrecords")

    for t_ind in range(len(val_input_names)):
        img = cv2.imread(val_input_names[t_ind],-1)
        label = cv2.imread(val_output_names[t_ind],-1)
        label = helpers.one_hot_it(label=label, num_classes=2).astype('uint8')
        # print(img.shape,img.dtype,label.shape,label.dtype)
        img_raw = img.tobytes() #将图片转化为原生bytes
        label_raw = label.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            "label_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
       features={
           'label_raw': tf.FixedLenFeature([], tf.string),
           'img_raw' : tf.FixedLenFeature([], tf.string),
       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label = tf.reshape(label, [256, 256, 2])
    label = tf.cast(label, tf.float32)
    return img, label

def reverse_one_hot(x):
    indices = tf.argmax(x,-1)
    indices = tf.reshape(indices,(args.batch_size,256,256,1))
    indices = tf.cast(indices,'float32')
    return indices

# Load the data
print("Loading the data ...")
train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = prepare_data()

# create_record()
img, label = read_and_decode(args.dataset+"/train1.tfrecords")
z, gt = tf.train.shuffle_batch([img, label],batch_size=args.batch_size, capacity=200, min_after_dequeue=100)

# img_val, label_val = read_and_decode("valid1.tfrecords")
# z_val, gt_val = tf.train.shuffle_batch([img_val, label_val],batch_size=1, capacity=200, min_after_dequeue=100)

# z, gt = tf.train.batch([img, label],batch_size=4, capacity=20)

# print(len(train_input_names),len(train_output_names),len(train_output_weight_names))
# print(len(val_input_names),len(val_output_names),len(test_input_names),len(test_output_names))

class_names_list = helpers.get_class_list(os.path.join(args.dataset, "class_list.txt"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(class_names_list)


network = None
init_fn = None
print("Preparing the model ...")
# with tf.device('/gpu'):
#     with tf.name_scope('tower_%d' % args.gpu_ids) as scope:
# z = tf.placeholder(tf.float32,shape=[None,None,None,3],name='input')
# gt = tf.placeholder(tf.float32,shape=[None,None,None,num_classes],name='output')
# if args.is_balanced_weight or args.is_edge_weight:
#     weight = tf.placeholder(tf.float32,shape=[None,None,None],name='weight')

if args.model == "FC-DenseNet56" or args.model == "FC-DenseNet67" or args.model == "FC-DenseNet103" or args.model == "FC-DenseNet158" or args.model == "FC-DenseNet232":
    if args.is_BC:
        G_logist, G, G_var = build_fc_densenet(z, preset_model = args.model, num_classes=num_classes, is_bottneck=True, compression_rate=0.5)
    else:
        G_logist, G, G_var = build_fc_densenet(z, preset_model = args.model, num_classes=num_classes, is_bottneck=False, compression_rate=1)
    
k_t = tf.Variable(0., trainable=False, name='k_t')
step = tf.Variable(0, name='step', trainable=False)

g_lr = tf.Variable(1e-3, name='g_lr')
d_lr = tf.Variable(8e-5, name='d_lr')

g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, 1e-5), name='g_lr_update')
d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, 1e-5), name='d_lr_update')

##############
# 5. build Discriminator
##############
data_format='NHWC'

gt_1d = reverse_one_hot(gt)
G_1d = reverse_one_hot(G)
d_out, GG, xx, D_var = Discriminator_Product_small(z, gt_1d, G_1d, 128, data_format)
AE_G, AE_x = tf.split(d_out, 2)

optimizer = tf.train.AdamOptimizer
g_optimizer, d_optimizer = tf.train.RMSPropOptimizer(learning_rate=g_lr, decay=0.995), optimizer(d_lr)

seg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=G_logist, labels=gt, dim=-1))
# seg_l1_loss = tf.reduce_mean(tf.abs(G - gt))

d_loss_real = (1e-8+tf.reduce_sum(gt_1d*tf.abs(AE_x - xx)))/(tf.reduce_sum(gt_1d)+1e-4)
d_loss_fake = (1e-8+tf.reduce_sum(G_1d*tf.abs(AE_G - GG)))/(tf.reduce_sum(G_1d)+1e-4)

seg_weight=1
gamma=1
lambda_k=0.001

g_loss = (1-seg_weight)*d_loss_fake + seg_weight*seg_loss
d_loss = d_loss_real - k_t * g_loss

d_optim = d_optimizer.minimize(d_loss, var_list=D_var)
g_optim = g_optimizer.minimize(g_loss, global_step=step, var_list=G_var)

balance = gamma * d_loss_real - g_loss
measure = d_loss_real + tf.abs(balance)

with tf.control_dependencies([d_optim, g_optim]):
    k_update = tf.assign(
        k_t, tf.clip_by_value(k_t + lambda_k * balance, 0, 1))
print("Building model done")

def denorm_img(norm, data_format):
    return tf.clip_by_value(norm*255, 0, 255)

summary_op = tf.summary.merge([
    tf.summary.image("z", denorm_img(z, data_format)),
    tf.summary.image("gt", denorm_img(reverse_one_hot(gt), data_format)),
    tf.summary.image("G", denorm_img(reverse_one_hot(G), data_format)),
    tf.summary.image("GG", denorm_img(GG, data_format)),
    tf.summary.image("xx", denorm_img(xx, data_format)),
    tf.summary.image("AE_G", denorm_img(AE_G, data_format)),
    tf.summary.image("AE_x", denorm_img(AE_x, data_format)),
    tf.summary.scalar("loss/d_loss", d_loss),
    tf.summary.scalar("loss/d_loss_real", d_loss_real),
    tf.summary.scalar("loss/d_loss_fake", d_loss_fake),
    tf.summary.scalar("loss/g_loss", g_loss),
    tf.summary.scalar("loss/seg_loss", seg_loss),
    tf.summary.scalar("misc/measure", measure),
    tf.summary.scalar("misc/k_t", k_t),
    tf.summary.scalar("misc/d_lr", d_lr),
    tf.summary.scalar("misc/g_lr", g_lr),
    tf.summary.scalar("misc/balance", balance),
])

##############
# 6 session
##############
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

##################
#summary
##################
saver = tf.train.Saver(max_to_keep=100)
summary_writer = tf.summary.FileWriter('checkpoints_#%d'%(args.exp_id), sess.graph)

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

model_checkpoint_name = "checkpoints_#%d/latest_model.ckpt" % args.exp_id #_" + args.model + "_" + args.dataset + "
if args.continue_training or not args.is_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, 'results_BEGAN/checkpoints_#81/0001/model.ckpt')

avg_scores_per_epoch = []

print("Config model done")

if args.is_training:

    print("***** Begin training *****")
    f = open('loss_#%d.txt' % (args.exp_id),'w')
    print('loss wirte to loss_#%d.txt' % (args.exp_id))
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Epochs -->", args.num_epochs)
    print("Batch Size -->", args.batch_size)
    print("exp_id -->", args.exp_id)
    print("gpu_ids>", args.gpu_ids)
    print("is_BC -->", args.is_BC)
    print("is_balanced_weight -->", args.is_balanced_weight)
    print("is_edge_weight -->", args.is_edge_weight)

    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tBrightness Alteration -->", args.brightness)
    print("\tRotation -->", args.rotation)
    print("\tZooming -->", args.zoom)
    print("")

    avg_loss_per_epoch = []
    step = 0
    max_step = 500000
    lr_update_step = 10000
    measure_history = deque([0]*lr_update_step, lr_update_step)

    # Which validation images doe we want
    # val_indices = []
    # num_vals = min(args.num_val_images, len(val_input_names))
    # for i in range(num_vals):
    #     ind = random.randint(0, len(val_input_names) - 1)
    #     val_indices.append(ind)

    # Do the training here
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord, start=True)  #启动QueueRunner, 此时文件名队列已经进队。

    for epoch in range(0, args.num_epochs):

        current_losses = []

        cnt=0
        id_list = np.random.permutation(len(train_input_names))
        num_iters = int(np.floor(len(id_list) / args.batch_size))

        for i in range(num_iters):
            st=time.time()

            t1 = time.time() - st
            # Do the training
            fetch_dict = {
                "k_update": k_update,
                "seg_loss": seg_loss,
                "measure": measure,
            }
            if step % 100 == 0:
                fetch_dict.update({
                    "summary": summary_op,
                    "g_loss": g_loss,
                    "d_loss": d_loss,
                    "k_t": k_t,
                })

            result = sess.run(fetch_dict) #,feed_dict={z:input_image_batch, gt:output_image_batch})

            measure_cur = result['measure']
            measure_history.append(measure_cur)
            seg_cur = result['seg_loss']
            current_losses.append(seg_cur)
            if step % 100 == 0:
                summary_writer.add_summary(result['summary'], step)
                summary_writer.flush()

                g_loss_cur = result['g_loss']
                d_loss_cur = result['d_loss']
                k_t_cur = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
                  format(step, max_step, d_loss_cur, g_loss_cur, seg_cur, k_t_cur))
            step += 1
            cnt = cnt + args.batch_size
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Current = %.2f t1 = %.2f Time = %.2f"%(epoch,cnt,seg_cur,t1, time.time()-st)
                utils.LOG(string_print)
        if epoch % 5 == 5 - 1:
            sess.run([g_lr_update, d_lr_update])

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)

        string_print = "Training loss: Epoch = %d, Count = %d, Epoch Loss = %.2f, Epoch measure Loss = %.2f"%(epoch,cnt,mean_loss,np.mean(measure_history))
        utils.LOG(string_print)
        f.writelines(str(mean_loss)+'\n')
        f.flush()

        # Create directories if needed
        if not os.path.isdir("checkpoints_#%d/%04d"%(args.exp_id, epoch)):
            os.makedirs("checkpoints_#%d/%04d"%(args.exp_id,epoch))

        saver.save(sess,model_checkpoint_name)
        saver.save(sess,"checkpoints_#%d/%04d/model.ckpt"%(args.exp_id,epoch))


        target=open("checkpoints_#%d/%04d/val_scores.txt"%(args.exp_id,epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou %s\n" % (class_names_string))

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []


    #     # Do the validation on a small set of validation images
    #     for ind in val_indices:

    #         # input_image = np.expand_dims(np.float32(cv2.cvtColor(cv2.imread(val_input_names[ind],-1), cv2.COLOR_BGR2RGB)[:args.crop_height, :args.crop_width]),axis=0)/255.0
    #         input_image = np.expand_dims(np.float32(cv2.imread(val_input_names[ind],-1)[:args.crop_height, :args.crop_width]),axis=0)/255.0
    #         gt_map = cv2.imread(val_output_names[ind],-1)[:args.crop_height, :args.crop_width]
    #         # print('gt labels',np.unique(gt_map))
    #         st = time.time()

    #         output_image = sess.run(G,feed_dict={z:input_image})
    #         output_image = np.array(output_image[0,:,:,:])
    #         output_image = helpers.reverse_one_hot(output_image)
    #         out_vis_image = helpers.colour_code_segmentation(output_image)

    #         accuracy = utils.compute_avg_accuracy(output_image, gt_map)
    #         class_accuracies = utils.compute_class_accuracies(output_image, gt_map, num_classes)
    #         prec = utils.precision(output_image, gt_map)
    #         rec = utils.recall(output_image, gt_map)
    #         f1 = utils.f1score(output_image, gt_map)
    #         iou = utils.compute_mean_iou(output_image, gt_map)

    #         file_name = utils.filepath_to_name(val_input_names[ind])
    #         target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
    #         for item in class_accuracies:
    #             target.write(", %f"%(item))
    #         target.write("\n")

    #         scores_list.append(accuracy)
    #         class_scores_list.append(class_accuracies)
    #         precision_list.append(prec)
    #         recall_list.append(rec)
    #         f1_list.append(f1)
    #         iou_list.append(iou)

    #         file_name = os.path.basename(val_input_names[ind])
    #         file_name = os.path.splitext(file_name)[0]
    #         cv2.imwrite("checkpoints_#%d/%04d/%s_pred.png"%(args.exp_id,epoch, file_name),np.uint8(out_vis_image))
    #         cv2.imwrite("checkpoints_#%d/%04d/%s_gt.png"%(args.exp_id,epoch, file_name),np.uint8(gt_map))


    #     target.close()

    #     avg_score = np.mean(scores_list)
    #     class_avg_scores = np.mean(class_scores_list, axis=0)
    #     avg_scores_per_epoch.append(avg_score)
    #     avg_precision = np.mean(precision_list)
    #     avg_recall = np.mean(recall_list)
    #     avg_f1 = np.mean(f1_list)
    #     avg_iou = np.mean(iou_list)

    #     print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
    #     print("Average per class validation accuracies for epoch # %04d:"% (epoch))
    #     for index, item in enumerate(class_avg_scores):
    #         print("%s = %f" % (class_names_list[index], item))
    #     print("Validation precision = ", avg_precision)
    #     print("Validation recall = ", avg_recall)
    #     print("Validation F1 score = ", avg_f1)
    #     print("Validation IoU score = ", avg_iou)

    #     scores_list = []

    # f.close()
    # fig = plt.figure(figsize=(11,8))
    # ax1 = fig.add_subplot(111)


    # ax1.plot(range(args.num_epochs), avg_scores_per_epoch)
    # ax1.set_title("Average validation accuracy vs epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Avg. val. accuracy")


    # plt.savefig('accuracy_vs_epochs_#%d.png' % args.exp_id)

    # plt.clf()

    # ax1 = fig.add_subplot(111)


    # ax1.plot(range(args.num_epochs), avg_loss_per_epoch)
    # ax1.set_title("Average loss vs epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Current loss")

    # plt.savefig('loss_vs_epochs_#%d.png' % args.exp_id)
    
    coord.request_stop()
    coord.join(threads)
else:
    print("***** Begin testing *****")

    # Create directories if needed
    if not os.path.isdir("Test_#%d"%(args.exp_id)):
            os.makedirs("Test_#%d"%(args.exp_id))

    target=open("Test_#%d/test_scores.txt"%(args.exp_id),'w')
    target.write("test_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
    scores_list = []
    class_scores_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []

    # Run testing on ALL test images
    for ind in range(len(test_input_names)):
        sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
        sys.stdout.flush()

        input_image = np.expand_dims(np.float32(cv2.cvtColor(cv2.imread(test_input_names[ind],-1), cv2.COLOR_BGR2RGB)[:args.crop_height, :args.crop_width]),axis=0)/255.0
        st = time.time()
        output_image = sess.run(network,feed_dict={z:input_image})


        gt_map = cv2.imread(test_output_names[ind],-1)[:args.crop_height, :args.crop_width]

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        output_image = output_image[:,:,0]
        out_vis_image = helpers.colour_code_segmentation(output_image)

        accuracy = utils.compute_avg_accuracy(output_image, gt_map)
        class_accuracies = utils.compute_class_accuracies(output_image, gt_map)
        prec = utils.precision(output_image, gt_map)
        rec = utils.recall(output_image, gt_map)
        f1 = utils.f1score(output_image, gt_map)
        iou = utils.compute_mean_iou(output_image, gt_map)

        file_name = utils.filepath_to_name(test_input_names[ind])
        target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
        for item in class_accuracies:
            target.write(", %f"%(item))
        target.write("\n")

        scores_list.append(accuracy)
        class_scores_list.append(class_accuracies)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        iou_list.append(iou)

        gt_map = helpers.reverse_one_hot(helpers.one_hot_it(gt_map))
        gt_map = helpers.colour_code_segmentation(gt_map)

        cv2.imwrite("Test_#%d/%s_pred.png"%(args.exp_id, file_name),np.uint8(out_vis_image))
        cv2.imwrite("Test_#%d/%s_gt.png"%(args.exp_id, file_name),np.uint8(gt_map))


    target.close()

    avg_score = np.mean(scores_list)
    class_avg_scores = np.mean(class_scores_list, axis=0)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_iou = np.mean(iou_list)
    print("Average test accuracy = ", avg_score)
    print("Average per class test accuracies = \n")
    for index, item in enumerate(class_avg_scores):
        print("%s = %f" % (class_names_list[index], item))
    print("Average precision = ", avg_precision)
    print("Average recall = ", avg_recall)
    print("Average F1 score = ", avg_f1)
    print("Average mean IoU score = ", avg_iou)