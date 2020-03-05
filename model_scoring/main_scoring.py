"""
References:
    https://github.com/google-research/google-research/tree/master/micronet_challenge
    Apache License, Version 2.0 - Copyright (c) 2019 The Google Research Authors
    https://github.com/lukemelas/EfficientNet-PyTorch
    Apache License, Version 2.0 - Copyright (c) 2019 Luke Melas-Kyriazi

This document is the main document for counting math operations and number of scoring parameters for
quantized networks solving tasks for the NIPS MicroNet. Global parameters and hyperparameters are set here.

We assume 'same' padding with square images/conv kernels.
BatchNorm scales are not counted since they can be merged. Bias added for each BatchNorm applied on a layer's output.

This document contains the following functions/classes:

    - main()
            - count_convs(layer_type, modules, bias, activation=None, ternary=False)
            - count_fc(modules, bias, activation=None, ternary=False)

        * optionally validating the model
            - validate(model, loss, val_iterator)
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import PIL
from collections import OrderedDict
from model import micronet, image_micronet, EfficientNet, best_cifar_micronet, lenet5, resnet18, resnet20, resnet50

parser = argparse.ArgumentParser(description='Calculating scores according to MicroNet Challenge')
parser.add_argument('--cuda', default=True, action='store_true',
                    help='By default CUDA training on GPU is enabled')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='Disables CUDA training and maps data plus model to CPU')
parser.add_argument('--workers', default=4, type=int, metavar=4,
                    help='Number of data loading workers (default: 4 per GPU)')
parser.add_argument('--val-batch-size', type=int, default=256, metavar=256,
                    help='Batch size for validation (default: 128)')
parser.add_argument('--model', default='cifar_micronet', type=str, metavar='cifar_micronet',
                    help='Choose model name / net generator function from [cifar_micronet, image_micronet,'
                         'efficientnet-b1, lenet-5, efficientnet-b2, efficientnet-b3, efficientnet-b4, resnet18, '
                         'resnet20] '
                         '(default: cifar_micronet)')
parser.add_argument('--t-model', default='best_ternary_cifar_micronet.pt',
                    type=str, metavar='ternary_model.pt',
                    help='Name of the ternary model file in "./model_scoring/trained_t_models" directory. '
                         'Choose from [best_ternary_cifar_micronet.pt, best_ternary_imagenet_micronet.pt]'
                         ' (default: best_ternary_cifar_micronet.pt)')
parser.add_argument('--dw-multps', type=float, default=(1.4, 1.2), nargs='+', metavar='1.4 1.2',
                    help='Depth and width multipliers d and w (default 1.4 1.2)')
parser.add_argument('--phi', type=float, default=3.5, metavar=3.5,
                    help='Phi is the exponential scaling factor for depth and width multipliers (default: 3.5)')
parser.add_argument('--dataset', default='CIFAR100', type=str, metavar='cifar',
                    help='Dataset to use. Choose from [CIFAR100, ImageNet] (default: CIFAR100)')
parser.add_argument('--data-path', default='../data', type=str, metavar='/path',
                    help='Path to ImageNet data. CIFAR data will be downloaded to "../data" directory automatically '
                         'if the data-path argument is ignored')
parser.add_argument('--image-size', default=32, type=int, metavar=32,
                    help='Input image size. Choose from [32 for CIFAR, 128-600 for ImageNet] (default: 32)')
parser.add_argument('--no-eval', default=True, action='store_true',
                    help='By default no forward pass on validation data')
parser.add_argument('--eval', dest='no_eval', action='store_false',
                    help='Setting --eval flag will cause a forward pass on validation data and return accuracy')
parser.add_argument('--halfprecision', default=True, action='store_true',
                    help='By default model parameters and input data will be quantized from 32 to 16 bit')
parser.add_argument('--no-halfprecision', dest='halfprecision', action='store_false',
                    help='Setting this flag will cause that parameters and input data remain 32 bit')
parser.add_argument('--prune-threshold', default=1, type=float, metavar=1,
                    help='If different from 1 this threshold sets all depthwise conv weights in EfficientNet to zero '
                         'which are below threshold*max(abs(layer_weights)) (default: 1)')
parser.add_argument('--mul-bits', default=16, type=int, metavar=16,
                    help='Number of bits inputs represented for multiplication (default: 16)')
parser.add_argument('--add-bits', default=32, type=int, metavar=32,
                    help='Number of bits used for accumulator.(default: 32)')
parser.add_argument('--no-save', default=True, action='store_true',
                    help='By default bitmasks and ternary weights are not saved layerwise')
parser.add_argument('--save', dest='no_save', action='store_false',
                    help='Save bitmasks and ternary weights layerwise')
parser.add_argument('--no-pt-sparse', default=True, action='store_true',
                    help='By default the network will not be saved as PyTorch.Sparse object')
parser.add_argument('--pt-sparse', dest='no_pt_sparse', action='store_false',
                    help='Save network layerwise as PyTorch.Sparse object, --save flag required')

# Globally defining parser and device (CPU/GPU)
global args, use_cuda, device

args = parser.parse_args()
use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('chosen device: ', device)

def count_convs(layer_type, modules, in_size=None, bias=False, activation=None,
                ternary=False, tree=False, cer=False):
    '''
    Counts the number of parameters and FLOPs for 2D conv layers. For counting the EfficientNet this funtion is
    is applied to all types of convolutions (expand, depthwise, Squeeze-and-Excitation, project, stem and head).

    Parameters:
    -----------
        layer_type:
            EfficientNet consists of several convolutional layer types, known as "stem", "expand", "depthwise",
            "SE_reduce", "SE_expand" (Squeeze and Excitation), "project" and "head". Find information on the layer types
            in the EfficientNet paper.
        modules:
            Iterable object containing so-called layer modules of a neural network. The modules should contain further
            information about each individual layer, not only the layer weights (i.e. number of filters, padding,
            kernel information, stride, ...)
        bias:
            Set to True if given layer type has bias weights or BatchNorm
        activation:
            Activation function of the given layer type (swish, sigmoid, relu)
        ternary:
            Set True if the given layer type was quantized ternary to benefit from reduced math ops and param count

    Returns:
    --------
        Total number of parameters for conv layers including BatchNorm parameters.
        Scoring parameters  as specified by NeurIPS MicroNet challenge.
        Number of bytes required to represent the conv layers.
        Total number of mathematical operations (FLOP) executed by conv layers.
    '''

    # Initializing variables for counting math ops and parameters
    num_params = []
    num_nonzero_params = []

    params_bits = []
    num_scoring_params = []
    dead_out_channel_indices_per_layer = []
    dead_in_channel_indices_per_layer = []
    dead_out_channel_indices_per_layer.append([])

    num_bn_params = []

    flop_mults_conv = 0
    flop_adds_conv = 0

    flop_adds_bn = 0

    flop_mults_act = 0
    flop_adds_act = 0

    flop_mults_avg = 0
    flop_adds_avg = 0

    flop_adds_skip = 0

    flop_mults_scale = 0

    # image_size = EfficientNet.get_image_size(args.model)
    if in_size:
        image_size = in_size
        input_size_div = 1
    else:
        image_size = args.image_size
        input_size_div = 0.5

    # Iterating over modules
    for i, mod in enumerate(modules):

        # Square kernel expected
        assert mod.kernel_size[0] == mod.kernel_size[1]
        k_size = mod.kernel_size[0]

        # Square stride expected
        assert mod.stride[0] == mod.stride[1]
        stride = mod.stride[0]

        # Padding is zero in ternary layers, Identity()
        padding = mod.padding[0]
        if "efficientnet" in args.model:
            if k_size == 3:
                padding = 1
            elif k_size == 5:
                padding = 2
            elif k_size == 1:
                padding = 0

        # Number of input and output channels
        c_in = mod.in_channels
        c_out = mod.out_channels

        # Layer params
        weights = mod.weight

        if args.model == 'efficientnet-b4':
            # Input image size is halved (pooled) at the following indices
            if layer_type != 'expand_conv' and layer_type != 'conv_stem' and layer_type != 'conv_head' and \
                    i in [2, 6, 10, 22]:
                input_size_div /= 2
            elif layer_type == 'expand_conv' and i in [0, 4, 8, 20]:
                input_size_div /= 2
            # The very first convolution layer has full resolution input size
            elif layer_type == 'conv_stem':
                input_size_div *= 2
            # The final convolution layer has a 5 times pooled input size
            elif layer_type == 'conv_head':
                input_size_div /= 16

        elif args.model == 'efficientnet-b3':
            # Input image size is halved (pooled) at the following indices
            if layer_type != 'expand_conv' and layer_type != 'conv_stem' and layer_type != 'conv_head' and \
                    i in [2, 5, 8, 18]:
                input_size_div /= 2
            elif layer_type == 'expand_conv' and i in [0, 3, 6, 16]:
                input_size_div /= 2
            # The very first convolution layer has full resolution input size
            elif layer_type == 'conv_stem':
                input_size_div *= 2
            # The final convolution layer has a 5 times pooled input size
            elif layer_type == 'conv_head':
                input_size_div /= 16

        elif args.model == 'efficientnet-b1' or args.model == 'efficientnet-b2':
            # Input image size is halved (pooled) at the following indices
            if layer_type != 'expand_conv' and layer_type != 'conv_stem' and layer_type != 'conv_head' and \
                    i in [2, 5, 8, 16]:
                input_size_div /= 2
            elif layer_type == 'expand_conv' and i in [0, 3, 6, 14]:
                input_size_div /= 2
            # The final convolution layer has a 5 times pooled input size
            elif layer_type == 'conv_head':
                input_size_div /= 16

        elif "net_conv" in layer_type and stride == 2:
            input_size_div /= 2

        # Append dead out channel indices per layer
        dead_out_channel_indices_per_layer.append([out_c for out_c in range(c_out)
                                                   if weights[out_c, :, :, :].sum() == 0])

        if layer_type != 'depthwise_conv' and 'net_conv' not in layer_type:
            dead_in_channel_indices_per_layer.append([in_c for in_c in range(c_in)
                                                      if weights[:, in_c, :, :].sum() == 0])
        elif 'net_conv' in layer_type:
            # in this case we can delete the according in channels from the former conv layer's deleted out channels
            dead_in_channel_indices_per_layer.append(dead_out_channel_indices_per_layer[i])
        else:
            dead_in_channel_indices_per_layer.append([])

        num_dead_in_c = len(dead_in_channel_indices_per_layer[i])
        num_dead_out_c = len(dead_out_channel_indices_per_layer[i+1])

        if num_dead_in_c > 0 or num_dead_out_c > 0:

            # Remove in_channels
            if num_dead_in_c > 0:
                scoring_in_c = [in_c for in_c in range(c_in)
                                if in_c not in dead_in_channel_indices_per_layer[i]]
            else:
                scoring_in_c = [in_c for in_c in range(c_in)]

            # Remove dead out_channels of current layer
            if num_dead_out_c > 0:
                scoring_out_c = [out_c for out_c in range(c_out)
                                if out_c not in dead_out_channel_indices_per_layer[i+1]]
            else:
                scoring_out_c = [out_c for out_c in range(c_out)]

            c_in -= num_dead_in_c
            c_out -= num_dead_out_c

            # Number of conv weights w/ dead channels removed
            if layer_type != 'depthwise_conv':
                num_params_wo_dead_c = c_in * c_out * k_size ** 2
                num_params += [num_params_wo_dead_c]
            else:
                num_params_wo_dead_c = c_out * k_size ** 2
                num_params += [num_params_wo_dead_c]

            # Number of nonzero weights in ternary conv layer w/ dead channels removed
            if layer_type != 'depthwise_conv':
                num_nonzero_params_wo_dead_c = sum([weights[out_c, in_c, :, :][weights[out_c, in_c, :, :] != 0].numel()
                                                       for out_c in scoring_out_c
                                                       for in_c in scoring_in_c])
            else:
                num_nonzero_params_wo_dead_c = sum([weights[out_c, :, :, :][weights[out_c, :, :, :] != 0].numel()
                                                       for out_c in scoring_out_c])

            num_nonzero_params += [num_nonzero_params_wo_dead_c]

        else:
            # Number of ternary conv weights
            num_params += [weights.numel()]

            # Number of nonzero weights in ternary conv layer
            num_nonzero_params += [weights[weights != 0].numel()]

        # Original sparsity of layer w/o having dead channels removed
        layer_sparsity = 1 - (num_nonzero_params[i] / num_params[i])

        #if not ternary or layer_sparsity > 0.937 :
        param_count = num_params[i] * (1 - layer_sparsity)
        bit_mask = 0
        if layer_sparsity > 0:
            bit_mask = num_params[i] # 1 bit binary mask
        params_bits += [param_count * num_bits + bit_mask]
        # Half precision values count as 1/2 scoring param binary mask as 1/32th param
        micronet_scoring = param_count * num_bits / bits_base + bit_mask / bits_base
        num_scoring_params += [param_count * num_bits / bits_base + bit_mask / bits_base]

        if ternary:
            ternary_scoring_1 = np.ceil((2 * num_params[i] + 2 * num_bits) / bits_base)
            if ternary_scoring_1 < micronet_scoring:
                num_scoring_params[i] = ternary_scoring_1
                params_bits[i] = ternary_scoring_1 * 32

        if ternary and cer:
            CSR_scoring_params = np.ceil(calc_csr(c_out, num_nonzero_params[i]) / bits_base)
            CER_scoring_params = np.ceil(calc_cer(c_out, num_nonzero_params[i]) / bits_base)
            ternary_scoring_1 = np.ceil((2 * num_params[i] + 2 * num_bits) / bits_base)
            ternary_scoring_2 = np.ceil((num_params[i] + param_count + 2 * num_bits) / bits_base)

            compressed_formats = np.array([micronet_scoring, CSR_scoring_params, CER_scoring_params,
                                           ternary_scoring_1, ternary_scoring_2])
            argmin_compress = compressed_formats.argmin()

            num_scoring_params[i] = compressed_formats.min()
            params_bits[i] = compressed_formats.min() * 32

            if argmin_compress == 0:
                print('MicroNet scoring applied: {}, {}'.format(compressed_formats.min(), compressed_formats))

            elif argmin_compress == 1:
                print('CSR applied: {}, {}'.format(compressed_formats.min(), compressed_formats))

            elif argmin_compress == 2:
                print('CER applied: {}, {}'.format(compressed_formats.min(), compressed_formats))

            elif argmin_compress == 3:
                print('Ternary scoring scheme 1 applied: {}, {}'.format(compressed_formats.min(), compressed_formats))

            elif argmin_compress == 4:
                print('Ternary scoring scheme 2 applied: {}, {}'.format(compressed_formats.min(), compressed_formats))

        # Each application of the kernel can be thought as a dot product between
        # the flattened kernel and patches of the image.
        if layer_type == 'depthwise_conv':
            vector_length = (k_size * k_size) * (1 - layer_sparsity)
        else:
            vector_length = (k_size * k_size * c_in) * (1 - layer_sparsity)

        # Calculating the output size
        out_size = np.ceil(((np.ceil(image_size * input_size_div) + (2 * padding) - k_size) / 1) + 1)

        # Number of elements in the output is OUT_SIZE * OUT_SIZE * OUT_CHANNEL
        if layer_type == 'se_reduce_conv' or layer_type == 'se_expand_conv':
            n_output_elements = c_out
        else:
            n_output_elements = int(out_size ** 2 * c_out)

        # Each output is the product of a one dot product. Dot product of two vectors of size n needs
        # n multiplications and n - 1 additions.
        # In the ternary case we multiply only the two cluster center values with the output patches. Why this
        # simplification is legit is described in the attached write up and can also be found in arXiv:1612.0106
        if ternary:
            flop_mults_conv += 2 * n_output_elements
        else:
            flop_mults_conv += vector_length * n_output_elements

        # Additions for convolution / tree adder
        if ternary and tree:
            for kernel_idx in range(c_out):
                nne = vector_length - 1
                base_i = num_bits
                addition_bits = 0
                while (nne > 1):
                    count_i = nne // 2
                    addition_bits += count_i * base_i
                    nne = np.ceil(nne / 2)
                    base_i += 1
                    if base_i > 32:
                        base_i = 32
                if layer_type == 'se_reduce_conv' or layer_type == 'se_expand_conv':
                    flop_adds_conv += addition_bits
                else:
                    flop_adds_conv += addition_bits * (out_size ** 2)
        else:
            flop_adds_conv += (vector_length - 1) * n_output_elements * args.add_bits

        # Parameter and math ops counting for Bias / BatchNorm
        if bias:
            # for the CIFARnet building blocks two BatchNorm layers for the first conv layer
            if layer_type != 'cifarnet_conv' or i % 2 == 0:
                # For each OUTPUT channel we need a bias term (BatchNorm merge)
                num_bn_params += [c_out]
                # If we have bias we need one more addition per dot product.
                flop_adds_bn += n_output_elements
            else:
                # For each OUTPUT channel we need a bias term (BatchNorm merge)
                num_bn_params += [c_in + c_out]
                # If we have bias we need one more addition per dot product.
                flop_adds_bn += 2 * n_output_elements

        # Ops counting for applied activation function
        if activation == 'swish':
            # Swish activation function x / (1 + exp(-bx)) counts as 3 multiplications and 1 addition
            flop_mults_act += 3 * n_output_elements
            flop_adds_act += n_output_elements
        elif activation == 'sigmoid':
            # Sigmoid activation function exp(x) / (1 + exp(x)) counts as 2 multiplications and 1 addition
            flop_mults_act += 2 * n_output_elements
            flop_adds_act += n_output_elements
        elif activation == 'relu':
            # for the CIFARnet building blocks only one ReLU for the first conv layer
            if layer_type != 'cifarnet_conv' or i % 2 == 0:
                # ReLU activation function counts as 1 multiplication
                flop_mults_act += n_output_elements

        # SkipOp adds
        if 'net_conv' in layer_type and i % 2 != 0:
            flop_adds_skip = n_output_elements

        # Global average ops for conv_head layer
        if layer_type == 'conv_head':
            # For each output channel we will make a division.
            flop_mults_avg += c_out
            # We have to add values over spatial dimensions.
            flop_adds_avg += (out_size * out_size -1) * c_out

        # Global average ops for SE reduce layer
        if layer_type == 'se_reduce_conv':
            # For each output channel we will make a division.
            flop_mults_avg += c_in
            # We have to add values over spatial dimensions.
            flop_adds_avg += (out_size * out_size -1) * c_in

        # Multiplying SE expand output with depthwise_conv's output
        if layer_type == 'se_expand_conv':
            # Number of elements many multiplications.
            flop_mults_scale += out_size ** 2 * c_out

        print('{} {}, sparsity: {:.2f}, kernel {}x{}, output size '
              '{}x{}, in/out channels {}/{}, # dead channels {}/{}, num_params {} , '
              'original conv mults {} and conv adds {}'.format(
                                layer_type, i, layer_sparsity * 100, k_size, k_size,
                                #int(np.ceil(image_size * input_size_div)), int(np.ceil(image_size * input_size_div)),
                                int(out_size), int(out_size),
                                c_in, c_out,  num_dead_in_c, num_dead_out_c, num_params[i],
                                int(vector_length * n_output_elements),
                                int((vector_length - 1) * n_output_elements)))

    # Representing BN bits, number of BatchNorm weights times 16 bit each (half precision)
    bn_params_bits = sum(num_bn_params) * num_bits

    # Scoring params for half precision BN values (count as 1/2 scoring param)
    num_bn_scoring_params = sum(num_bn_params) * num_bits / bits_base

    # Total counts
    total_params = sum(num_params) + sum(num_bn_params)
    total_bits = sum(params_bits) + bn_params_bits
    total_scoring_params = sum(num_scoring_params) + num_bn_scoring_params

    # Total ops
    mul_bits = args.mul_bits
    add_bits = args.add_bits
    total_mults = flop_mults_conv * mul_bits / bits_base / 1e6
    total_mults += flop_mults_act * mul_bits / bits_base / 1e6
    total_mults += flop_mults_avg * mul_bits / bits_base / 1e6
    total_mults += flop_mults_scale * mul_bits / bits_base / 1e6

    #total_adds = flop_adds_conv * add_bits / bits_base / 1e6
    total_adds = flop_adds_conv / bits_base / 1e6
    total_adds += flop_adds_skip * add_bits / bits_base / 1e6
    total_adds += flop_adds_bn * add_bits / bits_base / 1e6
    total_adds += flop_adds_act * add_bits / bits_base / 1e6
    total_adds += flop_adds_avg * add_bits / bits_base / 1e6

    print('-----------------------------------------------------------------------------------------------------')
    print('Number of {} layer weights: {}'.format(layer_type, sum(num_params)))
    print('Bytes required to represent {} layer weights: {:.4f} MB'.format(layer_type, sum(params_bits) / 8 / 1e6))
    print('Number of scoring params for {} layers: {}'.format(layer_type, int(sum(num_scoring_params))))
    print('-----------------------------------------------------------------------------------------------------')
    print('Number of {} layer BatchNorm params: {}'.format(layer_type, sum(num_bn_params)))
    print('Bytes required to represent {} layer BatchNorm params: {:.4f} MB'.format(layer_type,
                                                                                        bn_params_bits / 8 / 1e6))
    print('Number of scoring params for BatchNorm layers: ', int(num_bn_scoring_params))
    print('-----------------------------------------------------------------------------------------------------')
    print('Total number of {} layer params: {}'.format(layer_type, total_params))
    print('Total number of Bytes required for {} layers: {:.4} MB'.format(layer_type, total_bits / 8 / 1e6))
    print('-----------------------------------------------------------------------------------------------------')
    print('Total number of scoring params: ', int(total_scoring_params))
    print('-----------------------------------------------------------------------------------------------------')
    print('Number of ADDs: {:.2f} \n'.format(total_adds))
    print('Number MULTs: {:.2f} \n'.format(total_mults))
    print('Total number of MFLOP: {:.2f} \n'.format(total_mults + total_adds))
    print('-----------------------------------------------------------------------------------------------------')

    return total_params, int(total_scoring_params), total_bits / 8 / 1e6, total_mults, total_adds

def count_fc(modules, bias, activation=None, ternary=False):
    '''
    Counts the number of parameters and FLOPs for fully connected layers. In the EfficientNet this funtion is only
    applied to the very last network layer.

    Parameters:
    -----------
        modules:
            Iterable object containing so-called layer modules of a neural network. The modules should contain further
            information about each individual layer, not only the layer weights (i.e. number of filters, padding,
            kernel information, stride, ...)
        bias:
            Set to True if given layer type has bias weights or BatchNorm
        activation:
            Activation function of the given layer type (swish, sigmoid, relu)
        ternary:
            Set True if the given layer type was quantized ternary to benefit from reduced math ops and param count

    Returns:
    --------
        Total number of parameters for conv layers including BatchNorm parameters.
        Scoring parameters  as specified by NeurIPS MicroNet challenge.
        Number of bytes required to represent the conv layers.
        Total number of mathematical operations (FLOP) executed by conv layers.
    '''

    # Initializing variables for counting math ops and parameters
    num_params = []
    num_nonzero_params = []
    params_bits = []
    num_scoring_params = []
    num_bias_params = []
    bias_params_bits = []
    num_bias_scoring_params =[]

    flop_mults = 0
    flop_adds = 0
    flop_adds_bias = 0
    flop_mults_act = 0
    flop_adds_act = 0

    print('-----------------------------------------------------------------------------------------------------')
    print('FINAL FULLY CONNECTED LAYER')
    print('-----------------------------------------------------------------------------------------------------')

    # First (conv) layer counting
    for i, mod in enumerate(modules):

        # Number of input and output features
        c_in = mod.in_features
        c_out = mod.out_features

        # Layer params
        weights = mod.weight

        # Number of fc weights.
        num_params += [weights.numel()]
        # Number of nonzero weights in ternary conv layer
        num_nonzero_params += [weights[weights != 0].numel()]

        # Sparsity of layer
        layer_sparsity = 1 - (num_nonzero_params[i] / num_params[i])

        if not ternary:
            param_count = num_params[i] * (1 - layer_sparsity)
            bit_mask = 0
            if layer_sparsity > 0:
                bit_mask = num_params[i] # 1 bit binary mask
            params_bits += [param_count * num_bits + bit_mask]
            # Half precision values count as 1/2 scoring param binary mask as 1/32th param
            num_scoring_params += [param_count * num_bits / bits_base + bit_mask / bits_base]
        else:
            # 2 * binary mask for w_n and w_p + the half precision values of w_n and w_p
            params_bits += [2 * num_params[i] + 2 * num_bits]
            # Bitmask elements count as 1/32 param and centroid values as 1/2 param
            num_scoring_params += [2 * num_params[i] / bits_base + 2 * num_bits / bits_base]

        # Dot product ops.
        flop_mults += c_in * (1 - layer_sparsity) * c_out
        # We have one less addition than the number of multiplications per output features.
        flop_adds += (c_in * (1 - layer_sparsity) - 1) * c_out

        if bias:
            # Number of BatchNorm weights the number of out_features.
            num_bias_params += [c_out]
            # Number of BatchNorm weights times 16 bit each.
            bias_params_bits += [num_bias_params[i] * num_bits]
            # Half precision values count as 1/2 scoring param.
            num_bias_scoring_params += [num_bias_params[i] * num_bits / bits_base]

            # Bias add ops.
            flop_adds_bias += c_out

        # Ops counting for applied activation function
        if activation == 'swish':
            # Swish activation function x / (1 + exp(-bx)) counts as 3 multiplications and 1 addition
            flop_mults_act += 3 * c_out
            flop_adds_act += c_out
        elif activation == 'sigmoid':
            # Sigmoid activation function exp(x) / (1 + exp(x)) counts as 2 multiplications and 1 addition
            flop_mults_act += 2 * c_out
            flop_adds_act += c_out
        elif activation == 'relu':
            # ReLU activation function counts as 1 multiplication
            flop_mults_act += c_out

        print('FC layer {}, sparsity {:.2f}, in/out features {}/{}, # params {}, '
              '# representing bits {}, mults {}, adds {}'.format(i, layer_sparsity * 100, c_in, c_out, num_params[i],
                                                                 int(params_bits[i]),
            int(c_in * (1 - layer_sparsity) * c_out), int((c_in * (1 - layer_sparsity) - 1) * c_out)))

    # Total number of params
    total_params = sum(num_params) + sum(num_bias_params)
    total_bits = sum(params_bits) + sum(bias_params_bits)
    total_scoring_params = sum(num_scoring_params) + sum(num_bias_scoring_params)

    # Total math ops
    mul_bits = args.mul_bits
    add_bits = args.add_bits
    total_mults = flop_mults * mul_bits / bits_base / 1e6
    total_mults += flop_mults_act * mul_bits / bits_base / 1e6
    total_adds = flop_adds * add_bits / bits_base / 1e6
    total_adds += flop_adds_bias * add_bits / bits_base / 1e6
    total_adds += flop_adds_act * add_bits / bits_base / 1e6

    print('-----------------------------------------------------------------------------------------------------')
    print('Number of weights in final fully connected layer: ', sum(num_params))
    print('Bytes required to represent fully connected layer: {:.4f} MB'.format(sum(params_bits) / 8 / 1e6))
    print('Number of scoring params for fully connected layer: ', int(sum(num_scoring_params)))
    print('-----------------------------------------------------------------------------------------------------')
    print('Number of Bias weights in fully connected layer: ', sum(num_bias_params))
    print('Bytes required to represent FC bias weights: {:.4f} MB'.format(sum(bias_params_bits) / 8 / 1e6))
    print('Number of scoring params for FC bias weights: ', int(sum(num_bias_scoring_params)))
    print('-----------------------------------------------------------------------------------------------------')
    print('Total number of params in final fully connected layer: ', total_params)
    print('Total number of Bytes required for final fully connected layer: {:.4} MB'.format(total_bits / 8 / 1e6))
    print('-----------------------------------------------------------------------------------------------------')
    print('Total number of scoring params: ', int(total_scoring_params))
    print('-----------------------------------------------------------------------------------------------------')
    print('Total number of MFLOP: {:.2f} \n'.format(total_mults + total_adds))
    print('-----------------------------------------------------------------------------------------------------')

    return total_params, int(total_scoring_params), total_bits / 8 / 1e6, total_mults , total_adds


def validate(model, loss, val_iterator):
    """
    Evaluating the model (no backward path / opimization step)

    Parameters:
    -----------
        model:
            The neural network model.
        loss:
            Loss function used for optimization (e.g. cross entropy, MSE, ...).
        val_iterator:
            PyTorch Dataloader for dataset which should be evaluated.

    Returns:
    --------
        Validation loss.
        Validation accuracy.
    """

    # Initializing parameters
    loss_value = 0.0
    accuracy = 0.0
    total_samples = 0

    model.eval()

    with torch.no_grad():

        for i, (data, labels) in enumerate(val_iterator):

            correct = 0
            total = 0

            if use_cuda:
                data = data.cuda()
                labels = labels.cuda(non_blocking=True)

            # Enabling half precision 32 bit -> 16 bit
            if args.halfprecision:
                # Half precision requires cuda
                if use_cuda:
                    data = data.half()
                else:
                    print('WARNING: Half precision operation not possible on CPU')
                    args.halfprecision = False

            n_batch_samples = labels.size()[0]
            logits = model(data)

            # Compute batch loss
            batch_loss = loss(logits, labels)

            # Compute batch accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            batch_accuracy = 100. * correct / total

            loss_value += batch_loss.float()*n_batch_samples
            accuracy += batch_accuracy*n_batch_samples
            total_samples += n_batch_samples

            if i % 50 == 0:
                print('{} validated images: loss {:.3f}, accuracy {:.3f} %'.format((i + 1) * len(labels),
                                                                            loss_value/total_samples,
                                                                            accuracy/total_samples))

        return loss_value/total_samples, accuracy/total_samples

def calc_csr(rows, nonzeros):
    csr_params = (nonzeros * (num_bits + 1)) + ((rows + 1) * num_bits) + (2 * num_bits)
    return csr_params

def calc_cer(rows, nonzeros):
    cer_params = (nonzeros * num_bits) + (2 * ((rows + 1) * num_bits)) + (2 * num_bits)
    return cer_params

def main():
    """
    --------------------------------------------- MAIN --------------------------------------------------------
    Loads the models and data plus executes the counting functions. As a result a summary is printed.

    """

    # Transfers model to device (GPU/CPU). Device is globally initialized.
    if args.model == 'cifar_micronet':
        if args.dataset == 'CIFAR100':
            classes = 100
        elif args.dataset == 'CIFAR10':
            classes = 10
        model = best_cifar_micronet(num_classes=classes).to(device)

    elif args.model == 'image_micronet':
        model = image_micronet(args.dw_multps[0]**args.phi, args.dw_multps[1]**args.phi).to(device)

    elif args.model == 'efficientnet-b1':
        print('Building EfficientNet-B1 ...')
        model = EfficientNet.efficientnet_b1().to(device)

    elif args.model == 'efficientnet-b2':
        print('Building EfficientNet-B2 ...')
        model = EfficientNet.efficientnet_b2().to(device)

    elif args.model == 'efficientnet-b3':
        print('Building EfficientNet-B3 ...')
        model = EfficientNet.efficientnet_b3().to(device)

    elif args.model == 'efficientnet-b4':
        print('Building EfficientNet-B4 ...')
        model = EfficientNet.efficientnet_b4().to(device)

    elif args.model == 'resnet18':
        model = resnet18().to(device)

    elif args.model == 'resnet20':
        model = resnet20().to(device)

    elif args.model == 'resnet50':
        model = resnet50().to(device)

    else:
        raise NotImplementedError('Undefined model name %s' % args.model)

    # Defining loss function and printing CUDA information (if available)
    if use_cuda:
        print("PyTorch version: ")
        print(torch.__version__)
        print("CUDA Version: ")
        print(torch.version.cuda)
        print("cuDNN version is: ")
        print(cudnn.version())
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # Loading pretrained model
    t_model_path = './model_scoring/trained_t_models/'
    if args.t_model:
        if os.path.isfile(t_model_path + args.t_model):
            print("=> loading ternary model '{}'".format(t_model_path + args.t_model))

            trained_fp_model = torch.load(t_model_path + args.t_model, map_location=device)

            # For checkpoints unpack the state dict
            if 'state_dict' in trained_fp_model:
                trained_fp_model = trained_fp_model['state_dict']
            # For models which were saved in DataParallel mode (multiple GPUs) with "module." prefix
            for name, tensor in trained_fp_model.items():
                if 'module.' in name:
                    new_state_dict = OrderedDict()
                    for n, t in trained_fp_model.items():
                        name = n[7:]  # remove `module.`
                        new_state_dict[name] = t
                    # Load state dict
                    model.load_state_dict(new_state_dict)
                    break
                else:
                    # Load state dict
                    model.load_state_dict(trained_fp_model)
                    break

            print("=> loaded model successfully '{}'".format(t_model_path + args.t_model))
        else:
            print("=> no model found at '{}'".format(t_model_path + args.t_model))
            raise NotImplementedError('Model dictionary %s not available' % args.t_model)

    # Pruning w/o retraining
    if args.prune_threshold != 1:
        pruned_state_dict = OrderedDict()
        for name, tensor in model.state_dict().items():
            if 'depthwise' in name:
                tensor[tensor.abs() < tensor.abs().max() * args.prune_threshold] = 0
            pruned_state_dict[name] = tensor
        model.load_state_dict(pruned_state_dict)

    # Defining number of bits per weight / math op
    global num_bits, bits_base

    # A 32-bit parameter counts as one parameter
    bits_base = 32

    # Enabling half precision 32 bit -> 16 bit for model and also input data (see validate function)
    if args.halfprecision:
        # Half precision requires cuda
        num_bits = 16
        if use_cuda:
            model.half()
        else:
            print('WARNING: Half precision operation not possible on CPU')
            args.halfprecision = False
    else:
        num_bits = 32

    # Evaluation of ternary network
    if not args.no_eval:

        # Dataloaders for CIFAR, ImageNet
        if args.dataset == 'CIFAR100':

            print('CIFAR100')

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

            kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(root=args.data_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), download=True),
                batch_size=args.val_batch_size, shuffle=False, **kwargs)

        elif args.dataset == 'ImageNet':

            print('ImageNet')

            valdir = os.path.join(args.data_path, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            if model.__class__.__name__ == 'EfficientNet' or 'efficientnet' in str(args.model):
                image_size = args.image_size
                print('validation image_size is ', image_size)
                val_dataset = datasets.ImageFolder(
                    valdir,
                    transforms.Compose([
                        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        normalize,
                    ]))

            else:
                print('validation image_size is 224')
                val_dataset = datasets.ImageFolder(
                    valdir,
                    transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.val_batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

        elif args.dataset == 'CIFAR10':

            print('Loading CIFAR-10 data ...')
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

            kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root=args.data_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), download=True),
                batch_size=args.val_batch_size, shuffle=False, **kwargs)

        # Evaluation
        model.eval()
        start_time = time.time()
        val_loss, val_accuracy = validate(model, criterion, val_loader)
        print('val loss {:.3f}, val acc {:.3f} %, '
              'elapsed time {:.2f} s'.format(val_loss, val_accuracy, time.time() - start_time))

    # All model parameters
    all_weights = [param for name, param in model.named_parameters()]

    # Calculating model sparsity
    total_params = 0
    zero_params = 0
    for layer in all_weights:
        total_params += layer.numel()
        zero_params += layer[layer == 0].numel()
    total_sparsity = zero_params / total_params

    # In this case the C10- or C100-MicroNet is counted
    if model.__class__.__name__ == 'MicroNet':

        # First layer
        conv_weights = [param for name, param in model.named_modules() if 'conv1' in name
                        and not '.conv' in name]
        params_conv, scoring_params_conv, bytes_conv, mults_conv, adds_conv = count_convs(
            layer_type='cifarnet_conv',
            modules=conv_weights,
            in_size=32,
            bias=True,
            activation='relu',
            ternary=False
        )

        # Quantized convolution layers
        t_conv_weights = [param for name, param in model.named_modules() if '.conv' in name]
        params_tconv, scoring_params_tconv, bytes_tconv, mults_tconv, adds_tconv = count_convs(
            layer_type='cifarnet_conv',
            modules=t_conv_weights,
            in_size=32,
            bias=True,
            activation='relu',
            ternary=True,
            tree=True,
            cer=True
        )

        # Fully connected layer
        fc_weights = [param for name, param in model.named_modules() if ('linear' in name)]
        params_fc, scoring_params_fc, bytes_fc, mults_fc, adds_fc = count_fc(
            fc_weights, bias=True, ternary=False)

        # Summing up counts
        params = params_conv + params_tconv + params_fc
        scoring_params = scoring_params_conv + scoring_params_tconv + scoring_params_fc
        bytes = bytes_conv + bytes_tconv + bytes_fc
        adds = adds_conv + + adds_tconv + adds_fc
        mults = mults_conv + mults_tconv + mults_fc
        flops = adds + mults

    # In this case the ResNets for CIFAR-10 are counted, i.e. ResNet-20, -32, -44 & -56
    if model.__class__.__name__ == 'ResNetcifar':
        # First layer
        conv_weights = [param for name, param in model.named_modules() if 'conv1' in name
                        and not '.conv' in name]
        params_conv, scoring_params_conv, bytes_conv, mults_conv, adds_conv = count_convs(
            layer_type='resnet_conv_stem',
            modules=conv_weights,
            in_size=32,
            bias=True,
            activation='relu',
            ternary=False
        )

        # Quantized convolution layers
        t_conv_weights = [param for name, param in model.named_modules() if '.conv' in name]
        params_tconv, scoring_params_tconv, bytes_tconv, mults_tconv, adds_tconv = count_convs(
            layer_type='cifar_resnet_conv',
            modules=t_conv_weights,
            in_size=32,
            bias=True,
            activation='relu',
            ternary=False,
            tree=False,
            cer=False
        )

        # Fully connected layer
        fc_weights = [param for name, param in model.named_modules() if ('linear' in name)]
        params_fc, scoring_params_fc, bytes_fc, mults_fc, adds_fc = count_fc(
            fc_weights, bias=True, ternary=False)

        # Summing up counts
        params = params_conv + params_tconv + params_fc
        scoring_params = scoring_params_conv + scoring_params_tconv + scoring_params_fc
        bytes = bytes_conv + bytes_tconv + bytes_fc
        adds = adds_conv + + adds_tconv + adds_fc
        mults = mults_conv + mults_tconv + mults_fc
        flops = adds + mults

    # In this case the ResNets for ImageNet are counted, i.e. ResNet-18, -34 & -50
    if model.__class__.__name__ == 'ResNet':

        # First layer
        conv_weights = [param for name, param in model.named_modules() if 'conv1' in name
                        and not '.conv' in name]
        params_conv, scoring_params_conv, bytes_conv, mults_conv, adds_conv = count_convs(
            layer_type='in_resnet_conv_stem',
            modules=conv_weights,
            in_size=224,
            bias=True,
            activation='relu',
            ternary=False
        )

        # Quantized convolution layers
        t_conv_weights = [param for name, param in model.named_modules() if '.conv' in name]
        params_tconv, scoring_params_tconv, bytes_tconv, mults_tconv, adds_tconv = count_convs(
            layer_type='in_resnet_conv',
            modules=t_conv_weights,
            in_size=56,
            bias=True,
            activation='relu',
            ternary=True,
            tree=True,
            cer=True
        )

        # Fully connected layer
        fc_weights = [param for name, param in model.named_modules() if ('fc' in name)]
        params_fc, scoring_params_fc, bytes_fc, mults_fc, adds_fc = count_fc(
            fc_weights, bias=True, ternary=False)

        # Summing up counts
        params = params_conv + params_tconv + params_fc
        scoring_params = scoring_params_conv + scoring_params_tconv + scoring_params_fc
        bytes = bytes_conv + bytes_tconv + bytes_fc
        adds = adds_conv + + adds_tconv + adds_fc
        mults = mults_conv + mults_tconv + mults_fc
        flops = adds + mults


    # In this case the EfficientNets for ImageNet are counted
    if 'efficientnet' in args.model:

        tern=True
        cer=False
        tree=False

        # Conv stem ##################################################################################################
        conv_stem = [param for name, param in model.named_modules() if ('_conv_stem' in name)
                            and not ('static_padding' in name)]
        params_convstem, scoring_params_convstem, bytes_convstem, mults_convstem, adds_convstem = count_convs(
            layer_type='conv_stem', modules=conv_stem, bias=True, activation='swish')

        # Blocks #####################################################################################################
        expand_conv = [param for name, param in model.named_modules() if ('_expand_conv' in name)
                            and not ('static_padding' in name)]
        params_exp_conv, scoring_params_exp_conv, bytes_exp_conv, mults_exp_conv, adds_exp_conv = count_convs(
            layer_type='expand_conv', modules=expand_conv, bias=True, activation='swish',
            ternary=tern,
            tree=tree,
            cer=cer)
        # ------------------------------------------------------------------------------------------------------------
        depthwise_conv = [param for name, param in model.named_modules() if ('_depthwise_conv' in name)
                            and not ('static_padding' in name)]
        params_dw_conv, scoring_params_dw_conv, bytes_dw_conv, mults_dw_conv, adds_dw_conv = count_convs(
            layer_type='depthwise_conv', modules=depthwise_conv, bias=True, activation='swish',
            cer=cer)
        # ------------------------------------------------------------------------------------------------------------
        se_reduce_conv = [param for name, param in model.named_modules() if ('_se_reduce' in name)
                            and not ('static_padding' in name)]
        params_sered_conv, scoring_params_sered_conv, bytes_sered_conv, mults_sered_conv, adds_sered_conv = count_convs(
            layer_type='se_reduce_conv', modules=se_reduce_conv, bias=True, activation='swish',
            cer=cer)
        # ------------------------------------------------------------------------------------------------------------
        se_expand_conv = [param for name, param in model.named_modules() if ('_se_expand' in name)
                            and not ('static_padding' in name)]
        params_seexp_conv, scoring_params_seexp_conv, bytes_seexp_conv, mults_seexp_conv, adds_seexp_conv = count_convs(
            layer_type='se_expand_conv', modules=se_expand_conv, bias=True, activation='sigmoid',
            cer=cer)
        # ------------------------------------------------------------------------------------------------------------
        project_conv = [param for name, param in model.named_modules() if ('_project_conv' in name)
                            and not ('static_padding' in name)]
        params_proj_conv, scoring_params_proj_conv, bytes_proj_conv, mults_proj_conv, adds_proj_conv = count_convs(
            layer_type='project_conv', modules=project_conv, bias=True,
            ternary=tern,
            tree=tree,
            cer=cer)

        # Conv head ##################################################################################################
        conv_head = [param for name, param in model.named_modules() if ('_conv_head' in name)
                            and not ('static_padding' in name)]
        params_convhead, scoring_params_convhead, bytes_convhead, mults_convhead, adds_convhead = count_convs(
            layer_type='conv_head', modules=conv_head, bias=True, activation='swish',
            ternary=tern,
            tree=tree,
            cer=cer)

        # Fully connected ############################################################################################
        fc_layer = [param for name, param in model.named_modules() if ('_fc' in name)
                            and not ('static_padding' in name)]
        params_fc, scoring_params_fc, bytes_fc, mults_fc, adds_fc = count_fc(fc_layer, bias=True)

        # Add ops for shortcuts ######################################################################################
        flop_adds_skip = 0
        # image_size = EfficientNet.get_image_size(args.model)
        image_size = args.image_size
        input_size_div = 0.5
        for i, block in enumerate(model._blocks):

            if args.model == 'efficientnet-b4':
                if i in [2, 6, 10, 22]:
                    input_size_div /= 2
            elif args.model == 'efficientnet-b3':
                if i in [2, 5, 8, 18]:
                    input_size_div /= 2
            elif args.model == 'efficientnet-b1' or args.model == 'efficientnet-b2':
                if i in [2, 5, 8, 16]:
                    input_size_div /= 2

            input_size = image_size * input_size_div

            if (block._block_args.id_skip
                and block._block_args.input_filters == block._block_args.output_filters
                and block._block_args.stride == 1):
                flop_adds_skip += (input_size * input_size_div) ** 2 * block._block_args.output_filters

        flop_adds_skip *= args.add_bits / bits_base / 1e6

        ##############################################################################################################

        # Summing up counts
        params = params_convstem + params_exp_conv + params_dw_conv + params_sered_conv + params_seexp_conv + \
                 params_proj_conv + params_convhead + params_fc
        scoring_params = scoring_params_convstem + scoring_params_exp_conv + scoring_params_dw_conv + \
                         scoring_params_sered_conv + scoring_params_seexp_conv + scoring_params_proj_conv + \
                         scoring_params_convhead + scoring_params_fc
        bytes = bytes_convstem + bytes_exp_conv + bytes_dw_conv + bytes_sered_conv + bytes_seexp_conv + \
                 bytes_proj_conv + bytes_convhead + bytes_fc
        adds = adds_convstem + adds_exp_conv + adds_dw_conv + adds_sered_conv + adds_seexp_conv + \
                 adds_proj_conv + adds_convhead + adds_fc + flop_adds_skip
        mults = mults_convstem + mults_exp_conv + mults_dw_conv + mults_sered_conv + mults_seexp_conv + \
                 mults_proj_conv + mults_convhead + mults_fc
        flops = adds + mults


    # Saving sparse matrices as PyTorch-COO type or as binary bitmasks plus cluster values
    if not args.no_save:

        indices = []
        values = []

        for enum, t_layer in enumerate(t_conv_weights):

            # PyTorch sparse COO formatw
            if not args.no_pt_sparse:

                t_layer_sparse = t_layer.to_sparse()
                torch.save({
                    'indices': t_layer_sparse.indices(),
                    'values': t_layer_sparse.values(),
                    'size': t_layer_sparse.size()
                }, './model_scoring/saved_layers/layer_' + str(enum) + '_sparse.pt')

            # Saving indices and centroid weight layerwise
            else:
                centroids = torch.unique(t_layer)
                value = torch.HalfTensor([centroids[0], centroids[2]])
                indice_wn = torch.BoolTensor((t_layer == centroids[0]).type(torch.bool))
                indice_wp = torch.BoolTensor((t_layer == centroids[2]).type(torch.bool))

                indices.append([indice_wn.numpy, indice_wp.numpy])
                values.append(value.numpy)
                torch.save({
                    'indices_wn': indice_wn,
                    'indices_wp': indice_wp,
                    'values': value,
                }, './model_scoring/saved_layers/layer_' + str(enum) + '_sparse.pt')


    # Printing summary and overall (normalized) score
    print('\n -----------------------------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------------------------')
    print('Total number of network params: ', params)
    print('Network sparsity: {:.2f}%'.format(total_sparsity * 100))
    if not args.no_eval:
        print('Network validation accuracy: {:.2f}%'.format(val_accuracy))
    print('Total number of network scoring params: ', int(scoring_params))
    print('Total number of Bytes required for network: {:.4} MB'.format(bytes))
    print('Total number of inference Madds: {:.2f}'.format(adds))
    print('Total number of inference Mmults: {:.2f}'.format(mults))
    print('Total number of inference MFLOP: {:.2f}'.format(flops))
    print('-----------------------------------------------------------------------------------------------------')

    if args.dataset == 'ImageNet':
        print('Normalization relative to MobileNetV2 with width 1.4 (6.9M parameters and 1170M math operations)')
        print('Score: {:.3}M/6.9M + {:.2f}M/1170M = {:.4f} + {:.4f}'.format(scoring_params / 1e6, flops,
                                                                      scoring_params / 1e6 / 6.9, flops / (1170)))
        print('-----------------------------------------------------------------------------------------------------')
        print('Total score for {} task: {}'.format(str(args.dataset), scoring_params / 1e6 / 6.9 + flops / (1170)))

    elif args.dataset == 'CIFAR100':
        print('Normalization relative to WideResNet-28-10 (36.5M parameters and 10.49B math operations)')
        print('Score: {:.3}M/36.5M + {:.2f}M/10.49B = {:.4f} + {:.4f}'.format(scoring_params / 1e6, flops,
                                                                              scoring_params / 1e6 / 36.5,
                                                                              flops / (10.49 * 1e3)))
        print('-----------------------------------------------------------------------------------------------------')
        print('Total score for {} task: {}'.format(str(args.dataset),
                                                   scoring_params / 1e6 / 36.5 + flops / (10.49 * 1e3)))

    elif args.dataset == 'CIFAR10':
        print('Normalization relative to WideResNet-28-10 (36.5M parameters and 10.49B math operations)')
        print('Score: {:.3}M/36.5M + {:.2f}M/10.49B = {:.4f} + {:.4f}'.format(scoring_params / 1e6, flops,
                                                                              scoring_params / 1e6 / 36.5,
                                                                              flops / (10.49 * 1e3)))
        print('-----------------------------------------------------------------------------------------------------')
        print('Total score for {} task: {}'.format(str(args.dataset),
                                                   scoring_params / 1e6 / 36.5 + flops / (10.49 * 1e3)))
    print('-----------------------------------------------------------------------------------------------------')


if __name__ == '__main__':

    t1_start = time.perf_counter()

    main()

    t1_stop = time.perf_counter()

    print("Elapsed time: %.2f [s]" % (t1_stop-t1_start))
