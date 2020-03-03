'''
References:
    https://github.com/TropComplique/trained-ternary-quantization
    MIT License - Copyright (c) 2017 Dan Antoshchenko
    https://github.com/uoguelph-mlrg/Cutout
    Educational Community License, Version 2.0 (ECL-2.0) - Copyright (c) 2019 Vithursan Thangarasa
    https://github.com/lukemelas/EfficientNet-PyTorch
    Apache License, Version 2.0 - Copyright (c) 2019 Luke Melas-Kyriazi
    https://github.com/akamaster/pytorch_resnet_cifar10
    Yerlan Idelbayev's ResNet implementation for CIFAR10/CIFAR100 in PyTorch

This file contains functions for building the MicroNet model (for CIFAR100 and ImageNet) and for EfficientNet.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from .efficientnet_utils import (get_same_padding_conv2d, relu_fn, drop_connect, round_filters,
                                 round_repeats, get_model_params, load_pretrained_weights,
                                 efficientnet_params)

__all__ = ['MicroNet', 'micronet', 'image_micronet', 'best_cifar_micronet',
            'EfficientNet', 'lenet5', 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'ResNetcifar',
           'resnet20', 'resnet50']

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        # init.xavier_normal_(m.weight)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, len_blocks, stride=1, k_size1=3, k_size2=3):
        super(BasicBlock, self).__init__()

        if k_size1 == 3:
            pad1 = 1
        elif k_size1 == 5:
            pad1 = 2

        if k_size2 == 3:
            pad2 = 1
        elif k_size2 == 5:
            pad2 = 2

        # CIFAR
        if len_blocks == 3:
            self.dropout = nn.Dropout(0.2)

        # ImageNet
        if len_blocks == 7:
            self.dropout = nn.Dropout(0.0)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=k_size1, stride=stride, padding=pad1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=k_size2, stride=1, padding=pad2, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(planes)
            )

    def forward(self, x):

        out = self.dropout(self.relu(self.bn2(self.conv1(self.bn1(x)))))
        out = self.conv2(out)
        out = self.bn3(out)
        out += self.shortcut(x)

        return out

class MicroNet(nn.Module):
    def __init__(self, block, num_blocks, w_multiplier=1, num_classes=100, k_size=3):
        super(MicroNet, self).__init__()

        if k_size == 3:
            pad = 1
        elif k_size == 5:
            pad = 2

        self.len_blocks = len(num_blocks)


        # CIFAR
        if self.len_blocks == 3:

            self.in_planes = rounding_filters(16, w_multiplier)
            self.conv1 = nn.Conv2d(3, rounding_filters(16, w_multiplier), kernel_size=k_size, stride=1, padding=pad,
                                    bias=False)
            self.bn1 = nn.BatchNorm2d(rounding_filters(16, w_multiplier))
            self.layer1 = self._make_layer(block, rounding_filters(16, w_multiplier), num_blocks[0], stride=1, layer=1)
            self.layer2 = self._make_layer(block, rounding_filters(32, w_multiplier), num_blocks[1], stride=2, layer=2)
            self.layer3 = self._make_layer(block, rounding_filters(64, w_multiplier), num_blocks[2], stride=2, layer=3)
            self.linear = nn.Linear(rounding_filters(64, w_multiplier), num_classes)

        # ImageNet
        if self.len_blocks == 7:

            num_classes = 1000

            self.in_planes = rounding_filters(32, w_multiplier)
            self.conv1 = nn.Conv2d(3, rounding_filters(32, w_multiplier), kernel_size=k_size, stride=2, padding=pad,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(rounding_filters(32, w_multiplier))

            self.layer1 = self._make_layer(block, rounding_filters(16, w_multiplier), num_blocks[0], stride=1, layer=1)
            self.layer2 = self._make_layer(block, rounding_filters(24, w_multiplier), num_blocks[1], stride=2, layer=2)
            self.layer3 = self._make_layer(block, rounding_filters(40, w_multiplier), num_blocks[2], stride=2, layer=3)
            self.layer4 = self._make_layer(block, rounding_filters(80, w_multiplier), num_blocks[3], stride=2, layer=4)
            self.layer5 = self._make_layer(block, rounding_filters(112, w_multiplier), num_blocks[4], stride=1, layer=5)
            self.layer6 = self._make_layer(block, rounding_filters(192, w_multiplier), num_blocks[5], stride=2, layer=6)
            self.layer7 = self._make_layer(block, rounding_filters(320, w_multiplier), num_blocks[6], stride=1, layer=7)

            self.conv2 = nn.Conv2d(rounding_filters(320, w_multiplier), rounding_filters(1280, w_multiplier),
                                   kernel_size=1, stride=1, bias=False)
            self.bn2 = nn.BatchNorm2d(rounding_filters(1280, w_multiplier))

            self.dropout = nn.Dropout(0.3)
            self.linear = nn.Linear(rounding_filters(1280, w_multiplier), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, layer=None):
        strides = [stride] + [1]*(num_blocks-1) # if stride==2 only first layer in block has stride 2
        layers = []
        for k, stride in enumerate(strides):

            # ImageNet
            if self.len_blocks == 7:
                kernelsize1 = 3
                kernelsize2 = 3

            elif self.len_blocks == 3:
                kernelsize1 = 3
                kernelsize2 = 3

            layers.append(block(self.in_planes, planes, self.len_blocks, stride, k_size1=kernelsize1,
                                k_size2=kernelsize2))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # if ImageNet
        if self.len_blocks == 7:
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            out = self.layer7(out)
            out = F.relu(self.bn2(self.conv2(out)))
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = self.linear(out)
        # if CIFAR
        elif self.len_blocks == 3:
            out = F.relu(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def rounding_filters(filters, w_multiplier):
    """ Calculate and round number of filters based on width multiplier. """
    if not w_multiplier:
        return filters
    divisor = 8
    filters *= w_multiplier
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def rounding_repeats(repeats, d_multiplier):
    """ Round number of filters based on depth multiplier. """
    if not d_multiplier:
        return repeats
    return int(math.ceil(d_multiplier * repeats))


def micronet(d_multiplier, w_multiplier, num_classes):
    num_blocks = rounding_repeats(7, d_multiplier)
    return MicroNet(BasicBlock, [num_blocks, num_blocks, num_blocks], w_multiplier, num_classes=num_classes)


def best_cifar_micronet(d_multiplier=1.4**3.5, w_multiplier=1.2**3.5, num_classes=100):
    num_blocks = rounding_repeats(7, d_multiplier)
    return MicroNet(BasicBlock, [num_blocks, num_blocks, num_blocks], w_multiplier, num_classes=num_classes)


def image_micronet(d_multiplier, w_multiplier):
    num_blocks0 = rounding_repeats(1, d_multiplier)
    num_blocks1 = rounding_repeats(2, d_multiplier)
    num_blocks2 = rounding_repeats(2, d_multiplier)
    num_blocks3 = rounding_repeats(3, d_multiplier)
    num_blocks4 = rounding_repeats(3, d_multiplier)
    num_blocks5 = rounding_repeats(4, d_multiplier)
    num_blocks6 = rounding_repeats(1, d_multiplier)
    return MicroNet(BasicBlock, [num_blocks0, num_blocks1, num_blocks2, num_blocks3, num_blocks4,
                                 num_blocks5, num_blocks6], w_multiplier)

'''
-------------------------------------------------------------------------------------------------------------------
EFFICIENTNET
-------------------------------------------------------------------------------------------------------------------
'''
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def efficientnet_b1(cls):
        cls._check_model_name_is_valid('efficientnet-b1')
        blocks_args, global_params = get_model_params('efficientnet-b1', override_params={'num_classes': 1000})
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def efficientnet_b2(cls):
        cls._check_model_name_is_valid('efficientnet-b2')
        blocks_args, global_params = get_model_params('efficientnet-b2', override_params={'num_classes': 1000})
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def efficientnet_b3(cls):
        cls._check_model_name_is_valid('efficientnet-b3')
        blocks_args, global_params = get_model_params('efficientnet-b3', override_params={'num_classes': 1000})
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def efficientnet_b4(cls):
        cls._check_model_name_is_valid('efficientnet-b4')
        blocks_args, global_params = get_model_params('efficientnet-b4', override_params={'num_classes': 1000})
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

'''
-------------------------------------------------------------------------------------------------------------------
LeNet-5
-------------------------------------------------------------------------------------------------------------------
'''

class LeNet5(nn.Module):

    def __init__(self, d_mult=None, w_mult=None):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.max_pool_1(F.relu(self.conv1(x)))
        x = self.max_pool_2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def lenet5(d_multiplier, w_multiplier):
    return LeNet5(d_multiplier, w_multiplier)


'''
-------------------------------------------------------------------------------------------------------------------
ResNets from PyTorch model zoo for ImageNet
-------------------------------------------------------------------------------------------------------------------
'''

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class RBasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(RBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('RBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in RBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, RBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', RBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', RBasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

'''
-------------------------------------------------------------------------------------------------------------------
ResNets from Yerlan Idelbayev for CIFAR-10
-------------------------------------------------------------------------------------------------------------------
'''

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlockcifar(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockcifar, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetcifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetcifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20():
    return ResNetcifar(BasicBlockcifar, [3, 3, 3])

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])