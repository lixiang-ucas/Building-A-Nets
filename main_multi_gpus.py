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

import matplotlib.pyplot as plt

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
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--is_training', type=str2bool, default=True, help='Whether we are training or testing')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=360, help='Height of input image to network')
parser.add_argument('--crop_width', type=int, default=480, help='Width of input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Width of input image to network')
parser.add_argument('--num_val_images', type=int, default=10, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=None, help='Whether to randomly zoom in for data augmentation')
parser.add_argument('--model', type=str, default="FC-DenseNet103", help='The model you are using. Currently supports: FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, FC-DenseNet158, FC-DenseNet232, HF-FCN, Encoder-Decoder, Encoder-Decoder-Skip, RefineNet-Res50, RefineNet-Res101, RefineNet-Res152, FRRN-A, FRRN-B, MobileUNet, MobileUNet-Skip, PSPNet-Res50, PSPNet-Res101, PSPNet-Res152, GCN-Res50, GCN-Res101, GCN-Res152, custom')
parser.add_argument('--exp_id', type=int, default=1, help='Number of experiments')
parser.add_argument('--gpu_ids', type=str, default=0, help='List of GPU device id')
parser.add_argument('--is_BC', type=str2bool, default=False, help='whegher to use balanced weight')
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
    if args.is_edge_weight:
        for file in os.listdir(dataset_dir + "/train_labels_weights"):
            cwd = os.getcwd()
            train_output_weight_names.append(cwd + "/" + dataset_dir + "/train_labels_weights/" + file)
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
    return train_input_names,train_output_names, train_output_weight_names, val_input_names, val_output_names, test_input_names, test_output_names

def average_losses(loss):
    tf.add_to_collection('losses', loss)
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses')
    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
    Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y, batch_w):
    for i in range(len(models)):
        if args.is_edge_weight:
            input, output, weight, _, _, _ = models[i]
        else:
            input, output, _, _, _ = models[i]
        start_pos = i * payload_per_gpu
        stop_pos = (i + 1) * payload_per_gpu
        inp_dict[input] = batch_x[start_pos:stop_pos]
        inp_dict[output] = batch_y[start_pos:stop_pos]
        if ((args.is_edge_weight) and (batch_w is not None)):
            inp_dict[weight] = batch_w[start_pos:stop_pos]
    return inp_dict

# Check if model is available
AVAILABLE_MODELS = ["FC-DenseNet56", "FC-DenseNet67", "FC-DenseNet103", "FC-DenseNet158", "FC-DenseNet232", 
                    "Encoder-Decoder", "Encoder-Decoder-Skip", 
                    "RefineNet-Res101", "RefineNet-Res152", "HF-FCN", "custom"]
if args.model not in AVAILABLE_MODELS:
    print("Error: given model is not available. Try these:")
    print(AVAILABLE_MODELS)
    print("Now exiting ...")
    sys.exit()

# Load the data
print("Loading the data ...")
train_input_names, train_output_names, train_output_weight_names, val_input_names, val_output_names, test_input_names, test_output_names = prepare_data()

print(len(train_input_names),len(train_output_names),len(train_output_weight_names))
print(len(val_input_names),len(val_output_names),len(test_input_names),len(test_output_names))

class_names_list = helpers.get_class_list(os.path.join(args.dataset, "class_list.txt"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(class_names_list)

if args.is_balanced_weight:
    b_weight = utils.median_frequency_balancing(args.dataset + "/train_labels/", num_classes)

network = None
init_fn = None
print("Preparing the model ...")
opt = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.995)#.minimize(loss, var_list=[var for var in tf.trainable_variables()])
gpu_ids = args.gpu_ids.split(',')
print('build model...')
print('build model on gpu tower...')
models = []
for gpu_id in gpu_ids:
    gpu_id = int(gpu_id)
    with tf.device('/gpu:%d' % gpu_id):
        print('using tower:%d...'% gpu_id)
        with tf.name_scope('tower_%d' % gpu_id):
            with tf.variable_scope('gpu_variables', reuse=gpu_id>0):
                input = tf.placeholder(tf.float32,shape=[None,None,None,3])
                output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
                if args.is_balanced_weight or args.is_edge_weight:
                    weight = tf.placeholder(tf.float32,shape=[None,None,None])

                if args.model == "FC-DenseNet56" or args.model == "FC-DenseNet67" or args.model == "FC-DenseNet103" or args.model == "FC-DenseNet158" or args.model == "FC-DenseNet232":
                    if args.is_BC:
                        network = build_fc_densenet(input, preset_model = args.model, num_classes=num_classes, is_bottneck=1, compression_rate=0.5)
                    else:
                        network = build_fc_densenet(input, preset_model = args.model, num_classes=num_classes, is_bottneck=False, compression_rate=1)
                elif args.model == "RefineNet-Res50" or args.model == "RefineNet-Res101" or args.model == "RefineNet-Res152":
                    # RefineNet requires pre-trained ResNet weights
                    network, init_fn = build_refinenet(input, preset_model = args.model, num_classes=num_classes)
                elif args.model == "FRRN-A" or args.model == "FRRN-B":
                    network = build_frrn(input, preset_model = args.model, num_classes=num_classes)
                elif args.model == "Encoder-Decoder" or args.model == "Encoder-Decoder-Skip":
                    network = build_encoder_decoder(input, preset_model = args.model, num_classes=num_classes)
                elif args.model == "MobileUNet" or args.model == "MobileUNet-Skip":
                    network = build_mobile_unet(input, preset_model = args.model, num_classes=num_classes)
                elif args.model == "PSPNet-Res50" or args.model == "PSPNet-Res101" or args.model == "PSPNet-Res152":
                    # Image size is required for PSPNet
                    # PSPNet requires pre-trained ResNet weights
                    network, init_fn = build_pspnet(input, label_size=[args.crop_height, args.crop_width], preset_model = args.model, num_classes=num_classes)
                elif args.model == "GCN-Res50" or args.model == "GCN-Res101" or args.model == "GCN-Res152":
                    network, init_fn = build_gcn(input, preset_model = args.model, num_classes=num_classes)
                elif args.model == "custom":
                    network = build_custom(input, num_classes) 
                else:
                    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

                # Compute your (unweighted) softmax cross entropy loss
                if args.is_balanced_weight:
                    pixel_weight = b_weight*tf.argmax(input=output,dimension=3)+tf.argmin(input=output,dimension=3)
                    pixel_weight = tf.cast(pixel_weight, tf.float32)
                    loss = tf.reduce_mean(tf.multiply(pixel_weight*tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output)))
                elif args.is_edge_weight:
                    loss = tf.reduce_mean(tf.multiply(weight,tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output)))
                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output))
                    
                grads = opt.compute_gradients(loss)
                if args.is_edge_weight:
                    models.append((input,output,weight,network,loss,grads))
                else:
                    models.append((input,output,network,loss,grads))

print('build model on gpu tower done.')
print('reduce model on cpu...')
if args.is_edge_weight:
    _, _, _, tower_preds, tower_losses, tower_grads = zip(*models)
else:
    _, _, tower_preds, tower_losses, tower_grads = zip(*models)
aver_loss_op = tf.reduce_mean(tower_losses)
apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))
all_pred = tf.reshape(tf.stack(tower_preds, 0), [-1, args.crop_width, args.crop_height, num_classes])
print('reduce model on cpu done.')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

model_checkpoint_name = "checkpoints_#%d/latest_model.ckpt" % args.exp_id #_" + args.model + "_" + args.dataset + "
if args.continue_training or not args.is_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

avg_scores_per_epoch = []

if args.is_training:
    f = open('loss_#%d.txt' % (args.exp_id),'w')
    print("***** Begin training *****")
    print("exp_id -->", args.exp_id)
    print('loss wirte to loss_#%d.txt' % (args.exp_id))
    print("gpu_ids>", args.gpu_ids)
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Epochs -->", args.num_epochs)
    print("Batch Size -->", args.batch_size)
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

    # Which validation images doe we want
    val_indices = []
    num_vals = min(args.num_val_images, len(val_input_names))
    for i in range(num_vals):
        ind = random.randint(0, len(val_input_names) - 1)
        val_indices.append(ind)

    # Do the training here
    for epoch in range(0, args.num_epochs):

        current_losses = []

        cnt=0
        id_list = np.random.permutation(len(train_input_names))
        num_iters = int(np.floor(len(id_list) / args.batch_size))
        payload_per_gpu = args.batch_size/len(gpu_ids)

        for i in range(num_iters):
            st=time.time()
            
            input_image_batch = []
            output_image_batch = []
            pixel_weight_batch = [] 

            inp_dict = {}

            # Collect a batch of images
            for j in range(args.batch_size):
                index = i*args.batch_size + j
                id = id_list[index]
                input_image = cv2.cvtColor(cv2.imread(train_input_names[id],-1), cv2.COLOR_BGR2RGB)
                output_image = cv2.imread(train_output_names[id],-1)
                if args.is_edge_weight:
                    pixel_weight = cv2.imread(train_output_weight_names[id],-1)
                    # Data augmentation
                    input_image, output_image, pixel_weight = utils.random_crop(input_image, output_image, pixel_weight, args.crop_height, args.crop_width)
                else:
                    input_image, output_image = utils.random_crop(input_image, output_image, None, args.crop_height, args.crop_width)

                if args.h_flip and random.randint(0,1):
                    input_image = cv2.flip(input_image, 1)
                    output_image = cv2.flip(output_image, 1)
                    if args.is_edge_weight:
                        pixel_weight = cv2.flip(pixel_weight, 1)
                if args.v_flip and random.randint(0,1):
                    input_image = cv2.flip(input_image, 0)
                    output_image = cv2.flip(output_image, 0)
                    if args.is_edge_weight:
                        pixel_weight = cv2.flip(pixel_weight, 0)
                if args.brightness:
                    factor = 1.0 + abs(random.gauss(mu=0.0, sigma=args.brightness))
                    if random.randint(0,1):
                        factor = 1.0/factor
                    table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                    input_image = cv2.LUT(input_image, table)
                if args.rotation:
                    angle = args.rotation
                else:
                    angle = 0.0
                if args.zoom:
                    scale = args.zoom
                else:
                    scale = 1.0
                if args.rotation or args.zoom:
                    M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, scale)
                    input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]))
                    output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]))
                    if args.is_edge_weight:
                        pixel_weight = cv2.warpAffine(pixel_weight, M, (pixel_weight.shape[1], pixel_weight.shape[0]))

                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(helpers.one_hot_it(label=output_image, num_classes=num_classes))
                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))
                if args.is_edge_weight:
                    pixel_weight_batch.append(pixel_weight[np.newaxis,:,:])

            if args.batch_size == 1:
                input_image_batch = input_image_batch[0]
                output_image_batch = output_image_batch[0]
                if args.is_edge_weight:
                    pixel_weight_batch = pixel_weight_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
                if args.is_edge_weight:
                    pixel_weight_batch = np.squeeze(np.stack(pixel_weight_batch, axis=1))
                    # pixel_weight_batch = np.expand_dims(pixel_weight_batch, axis=3)

            # Do the training
            if args.is_edge_weight:
                inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, input_image_batch, output_image_batch, pixel_weight_batch)
            else:
                inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, input_image_batch, output_image_batch, None)
            _, current = sess.run([apply_gradient_op, aver_loss_op], inp_dict)

            current_losses.append(current)
            cnt = cnt + args.batch_size
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Current = %.2f Time = %.2f"%(epoch,cnt,current,time.time()-st)
                utils.LOG(string_print)

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)

        string_print = "Training loss: Epoch = %d Count = %d Epoch Loss = %.2f"%(epoch,cnt,mean_loss)
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

        # Do the validation on a small set of validation images
        total_batch = int(len(val_indices) / args.batch_size)
        val_payload_per_gpu = args.batch_size / len(gpu_ids)
        for i in range(total_batch):
            input_image_batch = []
            output_image_batch = []
            preds = None
            # Collect a batch of images
            for j in range(args.batch_size):
                ind = i*args.batch_size+j
                input_image = cv2.cvtColor(cv2.imread(val_input_names[ind],-1), cv2.COLOR_BGR2RGB)[:args.crop_height, :args.crop_width]/255.0
                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image = cv2.imread(val_output_names[ind],-1)[:args.crop_height, :args.crop_width]
                output_image = np.float32(helpers.one_hot_it(label=output_image, num_classes=num_classes))
                output_image_batch.append(np.expand_dims(output_image, axis=0))
            if args.batch_size == 1:
                input_image_batch = input_image_batch[0]
                output_image_batch = output_image_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
            inp_dict = feed_all_gpu({}, models, val_payload_per_gpu, input_image_batch, output_image_batch, None)
            batch_pred = sess.run([all_pred], inp_dict)
            # if preds is None:
            preds = batch_pred
            preds = np.stack(preds).reshape((args.batch_size,args.crop_width, args.crop_height,num_classes))
            # print('preds.shape:',preds.shape)
            for j in range(args.batch_size):
                gt = output_image_batch[j,:,:,:]
                output_image = preds[j,:,:,:] #np.squeeze(preds[j])
                output_image = helpers.reverse_one_hot(output_image)
                gt = helpers.reverse_one_hot(gt)
                out_vis_image = helpers.colour_code_segmentation(output_image)

                accuracy = utils.compute_avg_accuracy(output_image, gt)
                class_accuracies = utils.compute_class_accuracies(output_image, gt, num_classes)
                prec = utils.precision(output_image, gt)
                rec = utils.recall(output_image, gt)
                f1 = utils.f1score(output_image, gt)
                iou = utils.compute_mean_iou(output_image, gt)
            
                file_name = utils.filepath_to_name(val_input_names[ind])
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
                
                gt = helpers.reverse_one_hot(helpers.one_hot_it(gt))
                gt = helpers.colour_code_segmentation(gt)
     
                file_name = os.path.basename(val_input_names[ind])
                file_name = os.path.splitext(file_name)[0]
                cv2.imwrite("checkpoints_#%d/%04d/%s_pred.png"%(args.exp_id,epoch, file_name),np.uint8(out_vis_image))
                cv2.imwrite("checkpoints_#%d/%04d/%s_gt.png"%(args.exp_id,epoch, file_name),np.uint8(gt))

        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)

        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

        scores_list = []

    f.close()
    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)

    
    ax1.plot(range(args.num_epochs), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig('accuracy_vs_epochs_#%d.png' % args.exp_id)

    plt.clf()

    ax1 = fig.add_subplot(111)

    
    ax1.plot(range(args.num_epochs), avg_loss_per_epoch)
    ax1.set_title("Average loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs_#%d.png' % args.exp_id)

else:
    print("***** Begin testing *****")

    # Create directories if needed
    if not os.path.isdir("Test_#%d"%(args.exp_id)):
            os.makedirs("Test_#%d"%(args.exp_id))

    target=open("Test_#%d/test_scores.txt"%(args.exp_id),'w')
    target.write("test_name, avg_accuracy, precision, recall, f1 score, mean iou %s\n" % (class_names_string))
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
        output_image = sess.run(network,feed_dict={input:input_image})
        

        gt = cv2.imread(test_output_names[ind],-1)[:args.crop_height, :args.crop_width]

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        out_eval_image = output_image[:,:,0]
        out_vis_image = helpers.colour_code_segmentation(output_image)

        accuracy = utils.compute_avg_accuracy(out_eval_image, gt)
        class_accuracies = utils.compute_class_accuracies(out_eval_image, gt)
        prec = utils.precision(out_eval_image, gt)
        rec = utils.recall(out_eval_image, gt)
        f1 = utils.f1score(out_eval_image, gt)
        iou = utils.compute_mean_iou(out_eval_image, gt)
    
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
    
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt))
        gt = helpers.colour_code_segmentation(gt)

        cv2.imwrite("Test_#%d/%s_pred.png"%(args.exp_id, file_name),np.uint8(out_vis_image))
        cv2.imwrite("Test_#%d/%s_gt.png"%(args.exp_id, file_name),np.uint8(gt))


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