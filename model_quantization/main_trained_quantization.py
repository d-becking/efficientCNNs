"""
Trained Ternary Quantization with entropy controlled sparsity enhancement
implemented for our own compound scaled model, referred to as "MircoNet", solving CIFAR100
and for EfficientNet solving ImageNet

References:
    https://github.com/TropComplique/trained-ternary-quantization
    MIT License - Copyright (c) 2017 Dan Antoshchenko
    https://github.com/uoguelph-mlrg/Cutout
    Educational Community License, Version 2.0 (ECL-2.0) - Copyright (c) 2019 Vithursan Thangarasa
    https://github.com/lukemelas/EfficientNet-PyTorch
    Apache License, Version 2.0 - Copyright (c) 2019 Luke Melas-Kyriazi
    https://github.com/akamaster/pytorch_resnet_cifar10
    Yerlan Idelbayev's ResNet implementation for CIFAR10/CIFAR100 in PyTorch

This document is the main document for quantizing a pretrained MicroNet or EfficientNet. Global parameters
and hyperparameters are set here. Plus, the model, optimizers and dataloaders are initialized in this
main document. The function "grid_search" links to the quantizer.py document where the engine for the
entropy controlled ternary quantization can be found.

This document contains the following functions/classes:

    - main()
        - Cutout(object)
        - grid_search(train_loader, val_loader, model, loss_fct, lambda_max_divr, initial_c_divr)
            - define_optimizers(model, centroids)
            - k_means_ini(w_fp, num_centroids)
            - eval(model, loss_fct, val_loader)
"""

# Imports
import argparse
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
#from torchsummary import summary
import os
import PIL
from collections import OrderedDict
from .quantizer import quantize, train_quantization
from model import micronet, image_micronet, EfficientNet, best_cifar_micronet, lenet5, resnet18, resnet20, resnet50

# Initializing the parser and its arguments
parser = argparse.ArgumentParser(description='Entropy controlled ternary quantization')
parser.add_argument('--batch-size', type=int, default=256, metavar=256,
                    help='Batch size for training (default=256)')
parser.add_argument('--val-batch-size', type=int, default=512, metavar=512,
                    help='Batch size for validation (default=512)')
parser.add_argument('--epochs', type=int, default=20, metavar=20,
                    help='Number of epochs to train (default: 20)')
parser.add_argument('--retrain-epochs', type=int, default=15, metavar=15,
                    help='Number of epochs to retrain best model w/o changes in former assignment')
parser.add_argument('--lr', type=float, default=1e-4, metavar=1e-4,
                    help='Learning rate for full precision net update (default: 1e-4)')
parser.add_argument('--lr2', type=float, default=1e-5, metavar=1e-5,
                    help='Learning rate for centroid update (default: 1e-5)')
parser.add_argument('--cuda', default=True, action='store_true',
                    help='By default CUDA training on GPU is enabled')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='Disables CUDA training and maps data plus model to CPU')
parser.add_argument('--weight-decay', type=float, default=5e-6, metavar=5e-6,
                    help='Weight decay L2 (default 5e-6)')
parser.add_argument('--ini-c-divrs', type=float, default=[0.15, 0.3, 0.5], nargs='+', metavar='0.15 0.3 0.5 ...',
                    help='Multiplier(s) for the initial centroid values, '
                         'e.g. 0.3 * max(w) (default: 0.15, 0.3, 0.5)')
parser.add_argument('--lambda-max-divrs', type=float, default=[0.125, 0.15, 0.175], nargs='+',
                    metavar='0.125 0.15 0.175 ...',
                    help='Multiplier(s) for the maximum lambda per layer, '
                         'e.g. 0.15 * lambda_max (default: 0.125 0.15 0.175)')
parser.add_argument('--model', default='cifar_micronet', type=str, metavar='cifar_micronet',
                    help='Choose model name / net generator function from [cifar_micronet, image_micronet,'
                         'efficientnet-b1, lenet-5, efficientnet-b2, efficientnet-b3, efficientnet-b4, resnet18, '
                         'resnet20] '
                         '(default: cifar_micronet)')
parser.add_argument('--dw-multps', type=float, default=(1.4, 1.2), nargs='+', metavar='1.4 1.2',
                    help='Depth and width multipliers d and w (default 1.4 1.2)')
parser.add_argument('--phi', type=float, default=3.5, metavar=3.5,
                    help='Phi is the exponential scaling factor for depth and width multipliers (default: 3.5)')
parser.add_argument('--model-dict',
                    default='MicroNet_d14_w12_phi35_acc8146_params8_06m.th',
                    metavar='MicroNet_XY.pt',
                    help='''Choose name of pretrained full-precision model to quantize. It must be located in
                            "./model_quantization/trained_fp_models" directory. (default:
                            MicroNet_d14_w12_phi35_acc8146_params8_06m.th)''')
parser.add_argument('--dataset', default='CIFAR100', type=str, metavar='cifar',
                    help='Dataset to use. Choose from [CIFAR100, CIFAR10, ImageNet, MNIST] (default: CIFAR100)')
parser.add_argument('--image-size', default=200, type=int, metavar=200,
                    help='Input image size for ImageNet (default: 200)')
parser.add_argument('--data-path', default='../data', type=str, metavar='/path',
                    help='Path to ImageNet data. CIFAR data will be downloaded to "../data" directory automatically '
                         'if the data-path argument is ignored')
parser.add_argument('--workers', default=4, type=int, metavar=4,
                    help='Number of data loading workers (default: 4 per GPU)')
parser.add_argument('--resume', default='./model_quantization/saved_models/checkpoint.pt', type=str, metavar='/path',
                    help='Path to latest checkpoint')
parser.add_argument('--no-resume', dest='resume', action='store_false',
                    help='Do not resume from checkpoint')
parser.add_argument('--no-eval', default=True, action='store_true',
                    help='By default no forward pass on validation data')
parser.add_argument('--eval', dest='no_eval', action='store_false',
                    help='Setting --eval flag will cause a forward pass on validation data and return accuracy')
parser.add_argument('--prune-se', default=False, action='store_false',
                    help='By default no pruning applied')
parser.add_argument('--prune-fc', default=False, action='store_false',
                    help='By default no pruning applied')
parser.add_argument('--prune-dw', default=False, action='store_false',
                    help='By default no pruning applied')
parser.add_argument('--ec-pruning-se', dest='prune_se', action='store_true',
                    help='Setting --ec-pruning-se flag will cause entropy-constrained pruning of SE-layers')
parser.add_argument('--ec-pruning-fc', dest='prune_fc', action='store_true',
                    help='Setting --ec-pruning-fc flag will cause entropy-constrained pruning of FC-layer')
parser.add_argument('--ec-pruning-dw', dest='prune_dw', action='store_true',
                    help='Setting --ec-pruning-dw flag will cause entropy-constrained pruning of depthwise_conv-layers')
parser.add_argument('--centroids', default=0, type=int, metavar=0,
                    help='If different from 0, the given number of centroids (including 0 as a cluster center) will '
                         'be initialized using k-Means (default: 0)')
parser.add_argument('--plot-ternarization', default=False, type=bool, metavar='plot',
                    help='If True the ternarization process over epochs will be plotted (default: False)')
parser.add_argument('--slurm-save', type=str, metavar='/path',
                    help='Path where slurm saves job results.')


def grid_search(train_loader, val_loader, model, loss_fct, lambda_max_divr, initial_c_divr):
    """
    Sets up the training procedure with given hyperparameters (i.e. loading models, initial quantization and
    creating optimizers). After training a given number of epochs the best ternary network will be loaded again
    and will be further trained but without updating the quantization assignments but updating the centroid values
    and values of non-ternary layers such as 1st conv, final fully connected and batch_norm.

    Parameters:
    -----------
        train_loader:
            PyTorch Dataloader for given train dataset.
        val_loader:
            PyTorch Dataloader for given validation dataset.
        model:
            The neural network model.
        loss_fct:
            Loss function to use (e.g. cross entropy, MSE, ...).
        lambda_max_divr:
            The multiplier by which each layers lambda_max (i.e. the probability that all layer weights will be zero
            is close to 1) will be scaled.
        initial_c_divr:
            For the first optimization step all centroids (i.e. cluster centers [w_n, 0, w_p] in the ternary case)
            will get the values w_n  = w_min * initial_c_divr and respectively w_p = w_max * initial_c_divr.
    """

    # KERAS like summary of the model architecture
    if use_cuda:
        if args.dataset == 'CIFAR100' or args.dataset == 'CIFAR10' or args.dataset == 'MNIST':
            # summary(your_model, input_size=(channels, H, W), batch_size=-1, device="cuda")
            #summary(model, (3, 32, 32), batch_size=args.batch_size)
            print(model)

        elif args.dataset == 'ImageNet':
            #summary(model, (3, args.image_size, args.image_size), batch_size=args.batch_size)
            print(model)

    # Flag for entropy-constrained pruning instead of entropy-constrained trained ternarization (ECT2)
    if args.prune_se or args.prune_fc or args.prune_dw:
        pruning = True
        print("Entropy-constrained Pruning")
    else:
        pruning = False
        print("Entropy-constrained trained ternarization")

    # Optionally resume from a checkpoint
    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        stop_quantizing_flag = checkpoint['stop_quantizing_flag']
        initial_fp_weights = checkpoint['fp_update']
        initial_centroids = checkpoint['centroids']
        initial_assignment = checkpoint['assignment_list']

        # Optimizer for updating full_precision params
        opt_fp = optim.Adam(initial_fp_weights, lr=args.lr)
        # Creating optimizers for ternary weights and centroids
        optimizer, opt_sf = define_optimizers(model, initial_centroids)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        opt_fp.load_state_dict(checkpoint['opt_fp_state_dict'])
        opt_sf.load_state_dict(checkpoint['opt_sf_state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        del checkpoint


    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

        # If start training from scratch no frozen assignment fine-tuning possible
        stop_quantizing_flag = 0

        # Loading pretrained full-precision model to quantize
        trained_fp_model = torch.load('./model_quantization/trained_fp_models/' + args.model_dict,
                                      map_location=device)

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
        print("=> loaded '{}' ".format(str(args.model_dict)))

        # Parameters that will be quantized
        if model.__class__.__name__ == 'MicroNet' or 'micronet' in str(args.model) or 'resnet' in str(args.model):

            t_weights = [param for name, param in model.named_parameters()
                         if '.conv' in name]

            # image_micronet layer types which should be pruned subsequently to ternarization
            if args.prune_fc:
                t_weights = [param for name, param in model.named_parameters()
                             if (name == 'linear.weight' or name == 'conv2.weight')]

        elif model.__class__.__name__ == 'EfficientNet' or 'efficientnet' in str(args.model):

            # EfficientNet layer types which should be quantized ternary
            if not args.prune_se and not args.prune_fc and not args.prune_dw:
                t_weights = [param for name, param in model.named_parameters()
                            if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name)]

            # EfficientNet layer types which should be pruned subsequently to ternarization
            elif args.prune_se:
                print("Entropy-constrained Pruning of SE layers")
                t_weights = [param for name, param in model.named_parameters()
                             if ('_se_expand.weight' in name or '_se_reduce.weight' in name)]
            elif args.prune_fc:
                print("Entropy-constrained Pruning of FC layer")
                t_weights = [param for name, param in model.named_parameters()
                             if ('_fc.weight' in name)]

            elif args.prune_dw:
                print("Entropy-constrained Pruning of depthwise_conv-layers")
                t_weights = [param for name, param in model.named_parameters()
                             if ('_depthwise_conv' in name)]

        elif model.__class__.__name__ == 'LeNet5' or 'lenet' in str(args.model):
                t_weights = [param for name, param in model.named_parameters()
                                     if ('conv1.weight' in name or
                                     'conv2.weight' in name or
                                     'fc1.weight' in name or
                                     'fc2.weight' in name )]

        else:
            raise NotImplementedError('Undefined model class name %s' % model.__class__.__name__)

        # Copy of full precision weights to learn ternary assignments
        if model.__class__.__name__ == 'MicroNet' or 'micronet' in str(args.model) or 'resnet' in str(args.model):
            initial_fp_weights = [
                param.clone().detach().requires_grad_(True)
                for name, param in model.named_parameters()
                if '.conv' in name
            ]
            # image_micronet layer types which should be pruned subsequently to ternarization
            if args.prune_fc:
                initial_fp_weights = [
                    param.clone().detach().requires_grad_(True)
                    for name, param in model.named_parameters()
                    if (name == 'linear.weight' or name == 'conv2.weight')
                ]

        elif model.__class__.__name__ == 'EfficientNet' or 'efficientnet' in str(args.model):

            if not args.prune_se and not args.prune_fc and not args.prune_dw:
                initial_fp_weights = [
                    param.clone().detach().requires_grad_(True)
                    for name, param in model.named_parameters()
                    if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name)
                ]

            # EfficientNet layer types which should be pruned subsequently to ternarization
            elif args.prune_se:
                initial_fp_weights = [
                    param.clone().detach().requires_grad_(True)
                    for name, param in model.named_parameters()
                    if ('_se_expand.weight' in name or '_se_reduce.weight' in name)
                ]
            elif args.prune_fc:
                initial_fp_weights = [
                    param.clone().detach().requires_grad_(True)
                    for name, param in model.named_parameters()
                    if ('_fc.weight' in name)
                ]

            elif args.prune_dw:
                initial_fp_weights = [
                    param.clone().detach().requires_grad_(True)
                    for name, param in model.named_parameters()
                    if ('_depthwise_conv' in name)
                ]

        elif model.__class__.__name__ == 'LeNet5' or 'lenet' in str(args.model):
                initial_fp_weights = [
                    param.clone().detach().requires_grad_(True)
                    for name, param in model.named_parameters()
                                     if ('conv1.weight' in name or
                                     'conv2.weight' in name or
                                     'fc1.weight' in name or
                                     'fc2.weight' in name )
                ]
        else:
            raise NotImplementedError('Undefined model class name %s' % model.__class__.__name__)

        # Optimizer for updating full_precision params
        opt_fp = optim.Adam(initial_fp_weights, lr=args.lr)

        # Initial quantization
        initial_centroids = []
        initial_assignment = []
        for w, w_fp in zip(t_weights, initial_fp_weights):
            if args.centroids == 0:
                # Initial centroids by thresholding
                w_p_initial = w_fp.max().abs().item() * initial_c_divr
                w_n_initial = w_fp.min().abs().item() * initial_c_divr
                initial_centroids += [(w_p_initial, w_n_initial)]
            else:
                # Initial centroids by k-Means
                centroids = k_means_ini(w_fp, args.centroids)
                w_p_initial = centroids[2]
                w_n_initial = centroids[0]
                initial_centroids += [(w_p_initial, w_n_initial)]

            # Initial quantization (layerwise)
            w.data, _, assignment = quantize(w_fp.data, w_p_initial, w_n_initial, lambda_max_divr,
                                             Lambda=0.0, lambda_decay=1, cuda=use_cuda, pruning=pruning)
            initial_assignment += [assignment]

        # Creating optimizers for ternary weights and centroids
        optimizer, opt_sf = define_optimizers(model, initial_centroids)

    # If multipile GPUs are used
    if use_cuda and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # Transfers model to device (GPU/CPU).
    model.to(device)

    # Accuracy of pretrained net
    if not args.no_eval:
        eval(model, loss_fct, val_loader)

    ################################### TRAINING ######################################################################

    if stop_quantizing_flag == 0 and args.epochs != 0:
        print('Start ternary training')
        train_quantization(model, loss_fct, train_loader, val_loader, optimizer, opt_fp, opt_sf,
                           lambda_max_divr, initial_assignment, initial_c_divr, args.resume, use_cuda,
                           pruning=pruning, plot_flag=args.plot_ternarization, n_epochs=args.epochs,
                           slurm_save=args.slurm_save, stop_quantizing_flag=0)

    """
    loading the model with best ternary assignment (in terms of best validation accuracy) and
    further optimize the ternary cluster center values (w_n, w_p) with frozen spatial assignment
    """

    if args.retrain_epochs > 0:
        print('Start ternary training with fixed assignment')

        if args.slurm_save == None:
            best_ternary_dict = torch.load('./model_quantization/saved_models/Ternary_best_acc.pt', map_location=device)
        else:
            best_ternary_dict = torch.load(args.slurm_save + '/Ternary_best_acc.pt', map_location=device)

        # If multipile GPUs are used
        if use_cuda and torch.cuda.device_count() > 1:
            # For models which were saved in without Data.parallel "module." prefix
            parallel_state_dict = OrderedDict()
            for n, t in best_ternary_dict.items():
                name = 'module.' + n  # add `module.` prefix
                parallel_state_dict[name] = t
            model.load_state_dict(parallel_state_dict)
        else:
            model.load_state_dict(best_ternary_dict)

        if args.slurm_save == None:
            best_assignment = torch.load('./model_quantization/saved_models/assignments.pt', map_location=device)
            best_centroids = torch.load('./model_quantization/saved_models/centroids.pt', map_location=device)
        else:
            best_assignment = torch.load(args.slurm_save + '/assignments.pt', map_location=device)
            best_centroids = torch.load(args.slurm_save + '/centroids.pt', map_location=device)

        optimizer, opt_sf = define_optimizers(model, best_centroids)

        train_quantization(model, loss_fct, train_loader, val_loader, optimizer, opt_fp, opt_sf,
                           lambda_max_divr, best_assignment, initial_c_divr, args.resume, use_cuda, pruning=pruning,
                           plot_flag=args.plot_ternarization, n_epochs=args.retrain_epochs,
                            slurm_save=args.slurm_save, stop_quantizing_flag=1)

    ###################################################################################################################


def k_means_ini(w_fp, num_centroids):
    """
    Applies k-Means clustering to initialize the centroid values.
    Parameters:
    -----------
        w_fp:
            Full precision layer weights.
        num_centroids:
            Number of centroids for the given layer.

    Returns:
    --------
        centroids:
            Values for the initial centroids.
    """

    # Return evenly spaced centroid values over the interval [min_weight, max_weight]
    centroids = torch.linspace(w_fp.min().item(), w_fp.max().item(), num_centroids)

    # Setting the centroid closest to zero to zero
    centroids[centroids.abs() == centroids.abs().min()] = 0

    if use_cuda:
        centroids = centroids.cuda()

    finish = 0
    while finish != 1:

        centroids_old = centroids.clone().detach()

        # Measuring the squared distances from all weights to all centroids
        # Adding new axes to the centroid vector to make the subtraction/distance calculation with the layer's
        # weights shape possible
        G = torch.unsqueeze(centroids, 1)
        for i in range(1, w_fp.shape.__len__()):
            # Iteratively adding a new axis for the layers shape's length
            G = torch.unsqueeze(G, 1)

        distances = (G.sub(w_fp)) ** 2

        # Assigning weights to centroids with minimal distance
        assignment = torch.argmin(distances, dim=0)

        # Weight values assigned to each cluster
        assigned_values = [(assignment == c).float() * w_fp for c in range(num_centroids)]

        # Calculate centroid means (means of centroid assigned weight values)
        mean = [c.mean() for c in assigned_values]

        # List to tensor
        centroids = torch.stack(mean).detach()

        # Setting the centroid closest to zero to zero
        centroids[(centroids_old == 0).float().nonzero()] = 0

        # Stop while loop if the centroid update is marginal
        if (centroids - centroids_old).sum() <= 1e-6:
            finish = 1

    return centroids


def define_optimizers(model, centroids):
    """
    Splits the parameters which should be quantized ternary from those which should remain unquantized (BatchNorm,
    Bias, 1st and last layer of a network).
    Generates optimizers for ternary layers and centroids

    Parameters:
    -----------
        model:
            The neural network model
        centroids:
            Cluster center list of w_n, w_p per layer

    Returns:
    --------
        optimizer:
            Adam optimizer with two parameter groups: [0] for layers which will not be quantized (BatchNorm, Bias, ...)
            and [1] with layers that will be quantized ternary. Learning rate 1.
        opt_sf:
            Adam optimizer for the centroids (w_n, w_p) per layer.
            Learning rate 2 < learning rate 1 < approx. 0.1 or 0.01 * learning rate of pretraining
    """

    # Splitting not to quantize params

    if model.__class__.__name__ == 'MicroNet' or 'micronet' in str(args.model) or 'resnet' in str(args.model):

        # If first and last layer should not be quantized but only conv layers in Basic Blocks:
        params = [
            {'params': [param for name, param in model.named_parameters()
                        if not ('.conv' in name)], 'weight_decay': args.weight_decay},
            {'params': [param for name, param in model.named_parameters()
                        if '.conv' in name]}
        ]

        # After ternarization a third param group is appended. The second group contains layers which should be pruned,
        # the third param group contains the layers which were already ternarized. Their gradient will be set to zero
        # such that they will not change while pruning. The first group contains all remaining weights (depthwise conv,
        # batchnorm,...) and is updated as usual.
        if args.prune_fc:
            params = [
                {'params': [param for name, param in model.named_parameters()
                            if not ('.conv' in name or name == 'linear.weight' or name == 'conv2.weight')],
                                    'weight_decay': args.weight_decay},
                {'params': [param for name, param in model.named_parameters()
                            if (name == 'linear.weight' or name == 'conv2.weight')]},
                {'params': [param for name, param in model.named_parameters()
                            if ('.conv' in name)]}
            ]

    elif model.__class__.__name__ == 'EfficientNet' or 'efficientnet' in str(args.model):

        # EfficientNet layer types which should be quantized ternary
        if not args.prune_se and not args.prune_fc and not args.prune_dw:
            params = [
                {'params': [param for name, param in model.named_parameters()
                            if not ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name)],
                            'weight_decay': args.weight_decay},
                {'params': [param for name, param in model.named_parameters()
                            if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name)]}
            ]

        # After ternarization a third param group is appended. The second group contains layers which should be pruned,
        # the third param group contains the layers which were already ternarized. Their gradient will be set to zero
        # such that they will not change while pruning. The first group contains all remaining weights (depthwise conv,
        # batchnorm,...) and is updated as usual.

        #1
        elif args.prune_fc:
            params = [
                {'params': [param for name, param in model.named_parameters()
                            if not ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name or
                                    '_fc.weight' in name)],
                                    'weight_decay': args.weight_decay},
                {'params': [param for name, param in model.named_parameters()
                            if ('_fc.weight' in name)]},
                {'params': [param for name, param in model.named_parameters()
                            if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name)]}
            ]
        #2
        elif args.prune_se:
            params = [
                {'params': [param for name, param in model.named_parameters()
                            if not ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name or
                                    '_se_expand.weight' in name or '_se_reduce.weight' in name
                                    or '_fc.weight' in name)],
                                    'weight_decay': args.weight_decay},
                {'params': [param for name, param in model.named_parameters()
                            if ('_se_expand.weight' in name or '_se_reduce.weight' in name)]},
                {'params': [param for name, param in model.named_parameters()
                            if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name
                                or '_fc.weight' in name)]}
            ]
        #3
        elif args.prune_dw:
            params = [
                {'params': [param for name, param in model.named_parameters()
                            if not ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name or
                                    '_se_expand.weight' in name or '_se_reduce.weight' in name
                                    or '_fc.weight' in name or '_depthwise_conv' in name)],
                                    'weight_decay': args.weight_decay},
                {'params': [param for name, param in model.named_parameters()
                            if ('_depthwise_conv' in name )]},
                {'params': [param for name, param in model.named_parameters()
                            if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name
                                or '_fc.weight' in name or '_se_expand.weight' in name or '_se_reduce.weight' in name)]}
            ]

    elif model.__class__.__name__ == 'LeNet5' or 'lenet' in str(args.model):

        params = [
            {'params': [param for name, param in model.named_parameters()
                        if not ('conv1.weight' in name or
                        'conv2.weight' in name or
                        'fc1.weight' in name or
                        'fc2.weight' in name )], 'weight_decay': args.weight_decay},
            {'params': [param for name, param in model.named_parameters()
                        if ('conv1.weight' in name or
                        'conv2.weight' in name or
                        'fc1.weight' in name or
                        'fc2.weight' in name)]}
        ]
    else:
         raise NotImplementedError('Undefined model class name %s' % model.__class__.__name__)

    # Define model optimizer
    optimizer = optim.Adam(params, lr=args.lr)

    # Optimizer for updating centroids
    if use_cuda:
        opt_sf = optim.Adam([
            torch.tensor([w_p, w_n]).cuda().requires_grad_(True)
            for w_p, w_n in centroids
        ], lr=args.lr2)
    else:
        opt_sf = optim.Adam([
            torch.tensor([w_p, w_n]).requires_grad_(True)
            for w_p, w_n in centroids
        ], lr=args.lr2)

    return optimizer, opt_sf


def eval(model, loss_fct, val_loader):
    """
    Evaluating the model (no backward path / opimization step)

    Parameters:
    -----------
        model:
            The neural network model.
        loss_fct:
            Loss function used for optimization (e.g. cross entropy, MSE, ...).
        val_loader:
            PyTorch Dataloader for dataset which should be evaluated.
    """
    print('Evaluating the model...')
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        for data, target in val_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += loss_fct(output, target).item()
            # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.

    Parameters:
    -----------
        n_holes (int):
            Number of patches to cut out of each image.
        length (int):
            The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Parameters:
        -----------
            img (Tensor):
                Tensor image of size (C, H, W).
        Returns:
        --------
            Tensor:
                Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def main():
    """
    --------------------------------------------- MAIN --------------------------------------------------------

    Instantiates the model plus loss function and defines the dataloaders for several datasets including some
    data augmentation.
    Defines the grid for a grid search on lambda_max_divrs and initial_centroid_value_multipliers which both
    have a big influence on the sparsity (and respectively accuracy) of the resulting ternary networks.
    Starts grid search.
    """

    # Manual seed for reproducibility
    torch.manual_seed(363636)

    # Global instances
    global args, use_cuda, device
    # Instantiating the parser
    args = parser.parse_args()
    # Global CUDA flag
    use_cuda = args.cuda and torch.cuda.is_available()
    # Defining device and device's map locationo
    device = torch.device("cuda" if use_cuda else "cpu")
    print('chosen device: ', device)

    # Building the model
    if args.model == 'cifar_micronet':
        print('Building MicroNet for CIFAR with depth multiplier {} and width multiplier {} ...'.format(
            args.dw_multps[0] ** args.phi, args.dw_multps[1] ** args.phi))
        if args.dataset == 'CIFAR100':
            num_classes = 100
        elif args.dataset == 'CIFAR10':
            num_classes = 10
        model = micronet(args.dw_multps[0] ** args.phi, args.dw_multps[1] ** args.phi, num_classes)

    elif args.model == 'image_micronet':
        print('Building MicroNet for ImageNet with depth multiplier {} and width multiplier {} ...'.format(
            args.dw_multps[0] ** args.phi, args.dw_multps[1] ** args.phi))
        model = image_micronet(args.dw_multps[0] ** args.phi, args.dw_multps[1] ** args.phi)

    elif args.model == 'efficientnet-b1':
        print('Building EfficientNet-B1 ...')
        model = EfficientNet.efficientnet_b1()

    elif args.model == 'efficientnet-b2':
        print('Building EfficientNet-B2 ...')
        model = EfficientNet.efficientnet_b2()

    elif args.model == 'efficientnet-b3':
        print('Building EfficientNet-B3 ...')
        model = EfficientNet.efficientnet_b3()

    elif args.model == 'efficientnet-b4':
        print('Building EfficientNet-B4 ...')
        model = EfficientNet.efficientnet_b4()

    elif args.model == 'lenet-5':
        print('Building LeNet-5 with depth multiplier {} and width multiplier {} ...'.format(
            args.dw_multps[0] ** args.phi, args.dw_multps[1] ** args.phi))
        model = lenet5(d_multiplier=args.dw_multps[0] ** args.phi, w_multiplier=args.dw_multps[1] ** args.phi)

    elif args.model == "resnet18":
        model = resnet18()

    elif args.model == 'resnet20':
        model = resnet20()

    elif args.model == 'resnet50':
        model = resnet50()

    for name, param in model.named_parameters():
        print('\n', name)

    # Transfers model to device (GPU/CPU).
    model.to(device)

    # Defining loss function and printing CUDA information (if available)
    if use_cuda:
        print("PyTorch version: ")
        print(torch.__version__)
        print("CUDA Version: ")
        print(torch.version.cuda)
        print("cuDNN version is: ")
        print(cudnn.version())
        cudnn.benchmark = True
        loss_fct = nn.CrossEntropyLoss().cuda()
    else:
        loss_fct = nn.CrossEntropyLoss()

    # Dataloaders for CIFAR, ImageNet and MNIST
    if args.dataset == 'CIFAR100':

        print('Loading CIFAR-100 data ...')
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=args.data_path, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.075),
                transforms.ToTensor(),
                normalize,
                Cutout(n_holes=1, length=16),
            ]), download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=args.data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.val_batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'ImageNet':

        print('Loading ImageNet data ...')
        traindir = os.path.join(args.data_path, 'train')
        valdir = os.path.join(args.data_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        if model.__class__.__name__ == 'EfficientNet' or 'efficientnet' in str(args.model):
            image_size = EfficientNet.get_image_size(args.model)
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
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

    elif args.dataset == 'MNIST':

        kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.val_batch_size, shuffle=True, **kwargs)

    elif args.dataset == 'CIFAR10':

        print('Loading CIFAR-10 data ...')
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=args.data_path, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.075),
                transforms.ToTensor(),
                normalize,
                Cutout(n_holes=1, length=16),
            ]), download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=args.data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.val_batch_size, shuffle=False, **kwargs)

    else:
        raise NotImplementedError('Undefined dataset name %s' % args.dataset)


    # Gridsearch on dividers for lambda_max and initial cluster center values
    for initial_c_divr in args.ini_c_divrs:
        for lambda_max_divr in args.lambda_max_divrs:
            print('lambda_max_divr: {}, initial_c_divr: {}'.format(lambda_max_divr, initial_c_divr))
            if args.slurm_save == None:
                logfile = open('./model_quantization/logfiles/logfile.txt', 'a+')
            else:
                logfile = open(args.slurm_save + '/logfile.txt', 'a+')
            logfile.write('lambda_max_divr: {}, initial_c_divr: {}'.format(lambda_max_divr, initial_c_divr))
            grid_search(train_loader, val_loader, model, loss_fct, lambda_max_divr, initial_c_divr)


if __name__ == '__main__':
    t1_start = time.perf_counter()
    main()
    t1_stop = time.perf_counter()
    print("Elapsed time: %.2f [s]" % (t1_stop-t1_start))
