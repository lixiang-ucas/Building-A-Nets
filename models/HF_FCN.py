from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def preact_conv(inputs, n_filters, filter_size=[3, 3], dropout_p=0.2):
    """
    Basic pre-activation layer for DenseNets
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    Dropout (if dropout_p > 0) on the inputs
    """
    preact = tf.nn.relu(slim.batch_norm(inputs))
    conv = slim.conv2d(preact, n_filters, filter_size, activation_fn=None, normalizer_fn=None)
    if dropout_p != 0.0:
      conv = slim.dropout(conv, keep_prob=(1.0-dropout_p))
    return conv

def DenseBlock(stack, n_layers, growth_rate, dropout_p, scope=None):
  """
  DenseBlock for DenseNet and FC-DenseNet
  Arguments:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer
  Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
  """
  with tf.name_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
      # Compute new feature maps
      layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
      new_features.append(layer)
      # Stack new layer
      stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features


def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None):
  """
  Transition Down (TD) for FC-DenseNet
  Apply 1x1 BN + ReLU + conv then 2x2 max pooling
  """
  with tf.name_scope(scope) as sc:
    l = preact_conv(inputs, n_filters, filter_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
    return l


def TransitionUp(skip_connection, level, img_shape=np.array([1,256,256,3]), dropout_p=0.2, scope=None):
  """
  Transition Up for HF-FCN
  Performs upsampling on skip_connection by a factor 2
  """
  with tf.name_scope(scope) as sc:
    # Upsample

    l = preact_conv(skip_connection, 1, filter_size=[1, 1], dropout_p=dropout_p)
    if level>0:
      l = slim.conv2d_transpose(l, 1, kernel_size=[pow(2,level+1), pow(2,level+1)], stride=[pow(2,level), pow(2,level)])
    return l


def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def build_hf_fcn(inputs, preset_model='FC-DenseNet103', num_classes=12, n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_p=0.2, scope=None):
    """
    Builds the FC-DenseNet model
    Arguments:
      inputs: the input tensor
      preset_model: The model you want to use
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down = number of transition up
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
      dropout_p: dropout rate applied after each convolution (0. for not using)
    Returns:
      Fc-DenseNet model
    """

    if preset_model == 'FC-DenseNet56':
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      growth_rate=16
      n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      #####################
      # First Convolution #
      #####################
      # We perform a first convolution.
      stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv')

      n_filters = n_filters_first_conv
      #####################
      # Downsampling path #
      #####################

      skip_connection_list = []
      for i in range(n_pool):
        # Dense Block
        stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, scope='denseblock%d' % (i+1))
        n_filters += growth_rate * n_layers_per_block[i]
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, dropout_p, scope='transitiondown%d'%(i+1))

      #####################
      #     Bottleneck    #
      #####################

      # Dense Block
      # We will only upsample the new feature maps
      stack, _ = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + 1))

      skip_connection_list.append(stack)
      #######################
      #   Upsampling path   #
      #######################
      n_filters_keep = 1
      Fused = TransitionUp(skip_connection_list[0], 0, img_shape=inputs.shape, dropout_p=dropout_p, scope='P%d' % (i+1))
      for i in range(1,len(skip_connection_list)):
        feat_map = TransitionUp(skip_connection_list[i], i, img_shape=inputs.shape, dropout_p=dropout_p, scope='P%d' % (i+1))
        print feat_map.shape
        Fused = tf.concat([Fused, feat_map], axis=-1)

      #####################
      #      Softmax      #
      #####################
      net = slim.conv2d(Fused, num_classes, [1, 1], scope='logits')
      return net