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

This document is the main document for applying grid search on depth and width scaling factors for the MicroNet
architectures which can be found in ./model/model.

This document contains the following functions/classes:

    - main()
        - Cutout(object)
        - grid_search(train_loader, val_loader, criterion, alpha, beta)
            - train(train_loader, model, criterion, optimizer, epoch)
            - evaluate(model, loss, val_iterator)

'''

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#ssfrom torchsummary import summary
import numpy as np
import os
import PIL
from model import micronet, image_micronet, EfficientNet, best_cifar_micronet, lenet5


parser = argparse.ArgumentParser(description='Grid search for model compound scaling')
parser.add_argument('--workers', default=4, type=int, metavar='4',
                    help='Number of data loading workers (default: 4 per GPU)')
parser.add_argument('--epochs', default=200, type=int, metavar=200,
                    help='Number of epochs to run (default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar=0,
                    help='Start epoch, e.g. for restarts (default: 0)')
parser.add_argument('--batch-size', type=int, default=256, metavar=256,
                    help='Batch size for training (default=256)')
parser.add_argument('--val-batch-size', type=int, default=512, metavar=512,
                    help='Batch size for validation (default=512)')
parser.add_argument('--lr', default=0.1, type=float,
                    metavar=0.1, help='Initial learning rate (default 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar=0.9,
                    help='Momentum for SGD (default 0.9)')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    metavar=5e-4, help='Weight decay (default: 5e-4)')
parser.add_argument('--grid', type=float, default=(1.4, 1.2), nargs='+', metavar='1.4 1.2 1.2 1.3 ...',
                    help='Grid of alternating depth and width multipliers d and w (e.g.: d0, w0, d1, w1, ..., '
                         'dn, wn for n tuples)')
parser.add_argument('--phi', type=float, default=3.5, metavar=3.5,
                    help='Phi is the exponential scaling factor for width and depth multipliers (default: 3.5)')
parser.add_argument('--cuda', default=True, action='store_true',
                    help='By default CUDA training on GPU is enabled')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='Disables CUDA training and maps data plus model to CPU')
parser.add_argument('--dataset', default='CIFAR100', type=str, metavar='cifar',
                    help='Dataset to use. Choose from [CIFAR100, CIFAR10, MNIST, ImageNet] (default: CIFAR100)')
parser.add_argument('--image-size', default=32, type=int, metavar=32,
                    help='Input image size. Choose from [32 for CIFAR, 128-600 for ImageNet] (default: 32)')
parser.add_argument('--data-path', default='../data', type=str, metavar='/path',
                    help='Path to ImageNet data. CIFAR data will be downloaded to "../data" directory automatically '
                         'if the data-path argument is ignored')
parser.add_argument('--slurm-save', type=str, metavar='/path',
                    help='Path where slurm saves job results.')
parser.add_argument('--resume', default='./model_compound_scaling/saved_models/checkpoint.pt', type=str,
                    metavar='/path_cp',
                    help='Path to latest checkpoint. If it exists the train procedure will be resumed'
                         '(default: ./model_compound_scaling/saved_models/checkpoint.pt)')
parser.add_argument('--no-resume', dest='resume', action='store_false',
                    help='Do not resume from checkpoint')


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Training procedure.

    Parameters:
    -----------
        train_loader:
            PyTorch Dataloader for given train dataset.
        model:
            The neural network model.
        criterion:
            Loss function to use (e.g. cross entropy, MSE, ...).
        optimizer:
            Updating model parameters with Gradient Descent plus Nesterov momentum and weight decay.
        epoch:
            Current training epoch.

    Returns:
    --------
        running loss
        running accuracy
    """

    # Resetting variables for next epoch
    running_loss = 0.0
    running_accuracy = 0.0
    n_steps = 0

    # Iterating over training dataloader
    for input, target in train_loader:

        # Measure data loading time
        end = time.time()

        # Resetting variables for calculating current batch accuracy
        correct = 0
        total = 0

        # Map data to GPU if available
        if use_cuda:
            input, target = input.cuda(), target.cuda(non_blocking=True)

        # Forward pass
        output = model(input)

        # Benchmark mode can allocate large memory blocks during the very first forward to test algorithms
        if epoch == 0:
            torch.cuda.empty_cache()

        # Compute batch_loss
        batch_loss = criterion(output, target)

        # Compute accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        batch_accuracy = 100. * correct / total

        # Compute gradient and do optimization step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Summing up batch losses and accuracies over each step
        running_loss += batch_loss.float()
        running_accuracy += batch_accuracy
        n_steps += 1

        # Printing preliminary results
        if n_steps % 50 == 0:
            total_steps = int(len(train_loader.dataset) / train_loader.batch_size) + 1
            print('Epoch: {}, step: {}/{}, running_loss: {:.4f}, batch_acc: {:.4f}, running_acc: {:.4f}, '
                  'elapsed time: {:.2f} s, max_mem_alloc: {:.2f} GB, max_mem_cache {:.2f} GB'.format(
                epoch, n_steps, total_steps, running_loss / n_steps, batch_accuracy, running_accuracy / n_steps,
                                             (time.time() - end) * 100, torch.cuda.max_memory_allocated() / 1024 ** 3,
                                             torch.cuda.max_memory_cached() / 1024 ** 3))

        del output
        del batch_loss

    return running_loss / n_steps, running_accuracy / n_steps


def evaluate(model, loss, val_iterator):
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

    with torch.no_grad():

        # Iterating over validation dataloader
        for data, labels in val_iterator:

            # Resetting variables for calculating current batch accuracy
            correct = 0
            total = 0

            # Map data to GPU if available
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda(non_blocking=True)

            n_batch_samples = labels.size()[0]
            logits = model(data)

            # Compute batch loss
            batch_loss = loss(logits, labels)

            # Compute batch accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            batch_accuracy = 100. * correct / total

            # Summing up batch losses and accuracies over each step
            loss_value += batch_loss.float() * n_batch_samples
            accuracy += batch_accuracy * n_batch_samples
            total_samples += n_batch_samples

            del logits
            del batch_loss

        return loss_value / total_samples, accuracy / total_samples


def grid_search(train_loader, val_loader, criterion, alpha, beta):
    """
    Builds the model with given scaling factors, sets up optimizer and learning rate schedulers plus executes
    training and evaluation of the model. A checkpoint is created each epoch. Also the best model will be saved.

    Parameters:
    -----------
        train_loader:
            PyTorch Dataloader for given train dataset.
        val_loader:
            PyTorch Dataloader for given validation dataset.
        criterion:
            Loss function to use (e.g. cross entropy, MSE, ...).
        alpha:
            Scaling factor for model depth.
        beta:
            Scaling factor for model width.
    """

    # Initializing training variables
    best_acc = 0
    all_losses = []

    # Initializing log file
    if args.slurm_save == None:
        logfile = open('./model_compound_scaling/logfiles/logfile.txt', 'a+')
    else:
        logfile = open(args.slurm_save + '/logfile.txt', 'a+')
    logfile.write('depth multiplier: {}, width multiplier: {}\n'.format(alpha, beta))

    # Building the model
    if args.dataset == 'CIFAR100':
        model = micronet(d_multiplier=alpha, w_multiplier=beta, num_classes=100)

    elif args.dataset == 'CIFAR10':
        model = micronet(d_multiplier=alpha, w_multiplier=beta, num_classes=10)

    elif args.dataset == 'ImageNet':
        model = image_micronet(d_multiplier=alpha, w_multiplier=beta)

    elif args.dataset == 'MNIST':
        model = lenet5(d_multiplier=alpha, w_multiplier=beta)

    # If multipile GPUs are used
    if use_cuda and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Transfers model to device (GPU/CPU). Device is globally initialized.
    model.to(device)

    # Defining the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    # KERAS like summary of the model architecture
    # summary(your_model, input_size=(channels, H, W), batch_size=-1, device="cuda")
    if use_cuda:
        if args.dataset == 'CIFAR100' or args.dataset == 'CIFAR10':
            #summary(model, (3, 32, 32), batch_size=args.batch_size)
            print(model)

        elif args.dataset == 'ImageNet':
            #summary(model, (3, args.image_size, args.image_size), batch_size=args.batch_size)
            print(model)

        elif args.dataset == 'MNIST':
            #summary(model, (3, args.image_size, args.image_size), batch_size=args.batch_size)
            print(model)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            load_last_epoch = checkpoint['epoch']-1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            load_last_epoch = -1
    else:
        load_last_epoch = -1

    # Learning rate schedulers for cifar_micronet and imagenet_micronet
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max = args.epochs,
                                                           eta_min = 0,
                                                           last_epoch = load_last_epoch)

    # START TRAINING
    start_time = time.time()
    model.train()

    for epoch in range(args.start_epoch, args.epochs):

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        # Executing training process
        running_loss, running_accuracy = train(train_loader, model, criterion, optimizer, epoch)

        # Evaluation
        model.eval()
        val_loss, val_accuracy = evaluate(model, criterion, val_loader)

        # Logging the accuracies
        all_losses += [(epoch, running_loss, val_loss, running_accuracy, val_accuracy)]
        print('Epoch {0} running loss {1:.3f} val loss {2:.3f}  running acc {3:.3f} '
              'val acc{4:.3f}  time {5:.3f}'.format(*all_losses[-1], time.time() - start_time))
        logfile.write('Epoch {0} running loss {1:.3f} val loss {2:.3f}  running acc {3:.3f} '
              'val acc{4:.3f}  time {5:.3f}\n'.format(*all_losses[-1], time.time() - start_time))

        # Saving checkpoint
        if args.slurm_save == None:
            cp_savepath = './model_compound_scaling/saved_models/checkpoint.pt'
            best_savepath = './model_compound_scaling/saved_models/best_model.pt'
        else:
            cp_savepath = args.slurm_save + '/checkpoint.pt'
            best_savepath = args.slurm_save + '/best_model.pt'

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': val_accuracy,
            'lr': optimizer.param_groups[0]['lr']
        }, cp_savepath)

        # Make a lr scheduler step
        lr_scheduler.step()

        # Checking if current epoch yielded best validation accuracy
        is_best = val_accuracy > best_acc
        best_acc = max(val_accuracy, best_acc)

        # If so, saving best model state_dict
        if is_best and epoch > 0:
            torch.save(model.state_dict(), best_savepath)

        # Switch back to train mode
        model.train()
        start_time = time.time()


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
    Loads the data and executes the grid search on depth and width scaling factors.

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

    # Defining loss function and printing CUDA information (if available)
    if use_cuda:
        print("PyTorch version: ")
        print(torch.__version__)
        print("CUDA Version: ")
        print(torch.version.cuda)
        print("cuDNN version is: ")
        print(cudnn.version())
        cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # Dataloaders for CIFAR, ImageNet and MNIST
    if args.dataset == 'CIFAR100':

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

        traindir = os.path.join(args.data_path, 'train')
        valdir = os.path.join(args.data_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomAffine(degrees=15),
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                normalize,
            ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        image_size = args.image_size
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(image_size),
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

    # original grid = [(1.0, 1.0), (1.9, 1.0), (1.7, 1.1), (1.6, 1.1), (1.4, 1.2), (1.2, 1.3), (1.0, 1.4)]

    grid = [(args.grid[i], args.grid[i+1]) for i in range(0, len(args.grid), 2)]

    for coeff in grid:
        alpha = coeff[0] ** args.phi
        beta = coeff[1] ** args.phi
        grid_search(train_loader, val_loader, criterion, alpha, beta)


if __name__ == '__main__':
    t1_start = time.perf_counter()
    main()
    t1_stop = time.perf_counter()
    print("Elapsed time: %.2f [s]" % (t1_stop - t1_start))
