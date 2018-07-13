import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def Discriminator(z, gt, G, z_num, repeat_num, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      data_format=data_format,
                      activation_fn=tf.nn.elu,
                      #weights_initializer=tf.truncated_normal_initializer(0, 0.05),
                      weights_regularizer=slim.l2_regularizer(0.0005)
                           ):
            z = slim.conv2d(z, 16, 3)
            z = slim.conv2d(z, 64, 3)
            G = slim.conv2d(G, 64, 3)
            gt = slim.conv2d(gt, 64, 3)
            GG = tf.concat([z, G], 1)
            xx = tf.concat([z, gt], 1)
            print('GG.shape','xx.shape',GG.shape,xx.shape)
            x = tf.concat([GG,xx],0)
            # Encoder
            # x = slim.conv2d(x, hidden_num, 3)

            prev_channel_num = hidden_num
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3)
                # x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d(x, channel_num, 3)
                # x = slim.dropout(x, keep_prob=0.5)
                if idx < repeat_num - 1: #0-3
                    x = slim.conv2d(x, channel_num, 3, 2)
                    # x = slim.dropout(x, keep_prob=0.5)
                    # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
                    # x = slim.pool(x, [2, 2], stride=[2, 2], pooling_type='MAX', data_format='NCHW')
            down_size = 256/pow(2,repeat_num-1)
            x = tf.reshape(x, [-1, np.prod([down_size, down_size, channel_num])]) #8*8*(64*5)<---128/16,128/16,(64*5)
            z = x = slim.fully_connected(x, z_num, activation_fn=None)

            # Decoder
            num_output = int(np.prod([down_size, down_size, hidden_num]))
            x = slim.fully_connected(x, num_output, activation_fn=None)
            x = reshape(x, down_size, down_size, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1)
                # x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d(x, hidden_num, 3, 1)
                # x = slim.dropout(x, keep_prob=0.5)
                if idx < repeat_num - 1:
                    x = upscale(x, 2, data_format)
                    # x = slim.dropout(x, keep_prob=0.5)
                    # x = slim.conv2d_transpose(x, hidden_num, 3, 2)

            out = slim.conv2d(x, 128, 3)

    variables = tf.contrib.framework.get_variables(vs)
    return out, GG, xx, variables

def Discriminator_small(z, gt, G, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      data_format=data_format,
                      # activation_fn=tf.nn.elu,
                      #weights_initializer=tf.truncated_normal_initializer(0, 0.05),
                      # weights_regularizer=slim.l2_regularizer(0.0005)
                           ):
            GG = tf.concat([z, G], -1)
            xx = tf.concat([z, gt], -1)
            print('GG.shape','xx.shape',GG.shape,xx.shape)
            x = tf.concat([GG,xx],0)
            # Encoder-Decoder
            x = slim.conv2d(x, hidden_num, 3, 2)
            x = slim.conv2d(x, hidden_num, 3, 2)
            x = slim.conv2d(x, hidden_num, 3, 2)

            x = slim.conv2d_transpose(x, hidden_num, 3, 2)
            x = slim.conv2d_transpose(x, hidden_num, 3, 2)
            x = slim.conv2d_transpose(x, hidden_num, 3, 2)

            out = slim.conv2d(x, 5, 3)

    variables = tf.contrib.framework.get_variables(vs)
    return out, GG, xx, variables


def Discriminator_Product(z, gt, G, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      data_format=data_format,
                      # activation_fn=tf.nn.elu,
                      #weights_initializer=tf.truncated_normal_initializer(0, 0.05),
                      weights_regularizer=slim.l2_regularizer(0.0005)
                           ):
            GG = tf.multiply(z, G)
            xx = tf.multiply(z, gt)
            print('GG.shape','xx.shape',GG.shape,xx.shape)
            x = tf.concat([GG,xx],0)
            # Encoder
            # x = slim.conv2d(x, hidden_num, 3)

            prev_channel_num = hidden_num
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3)
                # x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d(x, channel_num, 3)
                # x = slim.dropout(x, keep_prob=0.5)
                if idx < repeat_num - 1: #0-3
                    x = slim.conv2d(x, channel_num, 3, 2)
                    # x = slim.dropout(x, keep_prob=0.5)
                    # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
                    # x = slim.pool(x, [2, 2], stride=[2, 2], pooling_type='MAX', data_format='NCHW')
            down_size = 256/pow(2,repeat_num-1)
            x = tf.reshape(x, [-1, np.prod([down_size, down_size, channel_num])]) #8*8*(64*5)<---128/16,128/16,(64*5)
            z = x = slim.fully_connected(x, z_num, activation_fn=None)

            # Decoder
            num_output = int(np.prod([down_size, down_size, hidden_num]))
            x = slim.fully_connected(x, num_output, activation_fn=None)
            x = reshape(x, down_size, down_size, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1)
                # x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d(x, hidden_num, 3, 1)
                # x = slim.dropout(x, keep_prob=0.5)
                if idx < repeat_num - 1:
                    x = upscale(x, 2, data_format)
                    # x = slim.dropout(x, keep_prob=0.5)
                    # x = slim.conv2d_transpose(x, hidden_num, 3, 2)

            out = slim.conv2d(x, 3, 3)

    variables = tf.contrib.framework.get_variables(vs)
    return out, GG, xx, variables

def Discriminator_Product_small(z, gt, G, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      data_format=data_format,
                      # activation_fn=tf.nn.elu,
                      #weights_initializer=tf.truncated_normal_initializer(0, 0.05),
                      weights_regularizer=slim.l2_regularizer(0.0005)
                           ):
            GG = tf.multiply(z, G)
            xx = tf.multiply(z, gt)
            print('GG.shape','xx.shape',GG.shape,xx.shape)
            x = tf.concat([GG,xx],0)

            # Encoder-Decoder
            x = slim.conv2d(x, hidden_num, 3, 2)
            x = slim.conv2d(x, hidden_num, 3, 2)
            x = slim.conv2d(x, hidden_num, 3, 2)

            x = slim.conv2d_transpose(x, hidden_num, 3, 2)
            x = slim.conv2d_transpose(x, hidden_num, 3, 2)
            x = slim.conv2d_transpose(x, hidden_num, 3, 2)

            out = slim.conv2d(x, 3, 3)

    variables = tf.contrib.framework.get_variables(vs)
    return out, GG, xx, variables

def Discriminator_Product_binary(z, gt, G, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      data_format=data_format,
                      # activation_fn=tf.nn.elu,
                      #weights_initializer=tf.truncated_normal_initializer(0, 0.05),
                      weights_regularizer=slim.l2_regularizer(0.0005)
                           ):
            GG = tf.multiply(z, G)
            xx = tf.multiply(z, gt)
            print('GG.shape','xx.shape',GG.shape,xx.shape)
            x = tf.concat([GG,xx],0)

            # Encoder-Decoder
            x = slim.conv2d(x, hidden_num, 3, 2)
            x = slim.conv2d(x, hidden_num*2, 3, 2)
            x = slim.conv2d(x, hidden_num*4, 3, 2)
            x = tf.reshape(x, [-1, np.prod([256/8, 256/8, hidden_num*4])])
            z = x = slim.fully_connected(x, 2, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    return out, GG, xx, variables