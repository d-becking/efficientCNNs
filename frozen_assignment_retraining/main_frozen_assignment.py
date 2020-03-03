import argparse
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import os
import PIL
from collections import OrderedDict
from model import micronet, image_micronet, EfficientNet, best_cifar_micronet, lenet5

# Initializing the parser and its arguments
parser = argparse.ArgumentParser(description='Entropy controlled ternary quantization')
parser.add_argument('--batch-size', type=int, default=256, metavar=256,
                    help='Batch size for training (default=256)')
parser.add_argument('--val-batch-size', type=int, default=512, metavar=512,
                    help='Batch size for validation (default=512)')
parser.add_argument('--epochs', type=int, default=20, metavar=20,
                    help='Number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4, metavar=1e-4,
                    help='Learning rate for full precision net update (default: 1e-4)')
parser.add_argument('--cuda', default=True, action='store_true',
                    help='By default CUDA training on GPU is enabled')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='Disables CUDA training and maps data plus model to CPU')
parser.add_argument('--weight-decay', type=float, default=5e-6, metavar=5e-6,
                    help='Weight decay L2 (default 5e-6)')
parser.add_argument('--model', default='cifar_micronet', type=str, metavar='cifar_micronet',
                    help='Choose model name / net generator function from [cifar_micronet, image_micronet,'
                         'efficientnet-b1, lenet-5, efficientnet-b2, efficientnet-b3, efficientnet-b4] '
                         '(default: cifar_micronet)')
parser.add_argument('--dw-multps', type=float, default=(1.4, 1.2), nargs='+', metavar='1.4 1.2',
                    help='Depth and width multipliers d and w (default 1.4 1.2)')
parser.add_argument('--phi', type=float, default=3.5, metavar=3.5,
                    help='Phi is the exponential scaling factor for depth and width multipliers (default: 3.5)')
parser.add_argument('--model-dict',
                    default='',
                    metavar='MicroNet_XY.pt',
                    help='''Choose name of pretrained full-precision model to quantize. It must be located in
                            "./model_quantization/trained_fp_models" directory. ''')
parser.add_argument('--dataset', default='CIFAR100', type=str, metavar='cifar',
                    help='Dataset to use. Choose from [CIFAR100, CIFAR10, ImageNet, MNIST] (default: CIFAR100)')
parser.add_argument('--image-size', default=224, type=int, metavar=224,
                    help='Input image size for ImageNet (default: 224)')
parser.add_argument('--data-path', default='../data', type=str, metavar='/path',
                    help='Path to ImageNet data. CIFAR data will be downloaded to "../data" directory automatically '
                         'if the data-path argument is ignored')
parser.add_argument('--workers', default=4, type=int, metavar=4,
                    help='Number of data loading workers (default: 4 per GPU)')
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
parser.add_argument('--slurm-save', type=str, metavar='/path',
                    help='If Slurm is used, path where slurm saves job results.')

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
        # Set gradients of quantized and pruned layer to zero to freeze their weight values
        for i in range(len(optimizer.param_groups[1]['params'])):
            optimizer.param_groups[1]['params'][i].grad.data.zero_()
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


def train_w_frozen_assignment(train_loader, val_loader, model, loss_fct):

    # Loading pretrained model
    quantized_model = torch.load('./frozen_assignment_retraining/quantized_models/' + args.model_dict,
                                  map_location=device)

    # For checkpoints unpack the state dict
    if 'state_dict' in quantized_model:
        quantized_model = quantized_model['state_dict']
    # For models which were saved in DataParallel mode (multiple GPUs) with "module." prefix
    for name, tensor in quantized_model.items():
        if 'module.' in name:
            new_state_dict = OrderedDict()
            for n, t in quantized_model.items():
                name = n[7:]  # remove `module.`
                new_state_dict[name] = t
            # Load state dict
            model.load_state_dict(new_state_dict)
            break
        else:
            # Load state dict
            model.load_state_dict(quantized_model)
            break

    del quantized_model

    # Layer types which were quantized ternary or pruned

    if model.__class__.__name__ == 'MicroNet' or 'micronet' in str(args.model):

        # If first and last layer should not be quantized but only conv layers in Basic Blocks:
        params = [
            {'params': [param for name, param in model.named_parameters()
                        if not ('.conv' in name)], 'weight_decay': args.weight_decay},
            {'params': [param for name, param in model.named_parameters()
                        if '.conv' in name]}
        ]

    elif model.__class__.__name__ == 'EfficientNet' or 'efficientnet' in str(args.model):

        # Assuming that first the FC layer was pruned, followed by the SE module and finally the depthwise layers

        if not args.prune_se and not args.prune_fc and not args.prune_dw:
            params = [
                {'params': [param for name, param in model.named_parameters()
                            if not ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name)],
                            'weight_decay': args.weight_decay},
                {'params': [param for name, param in model.named_parameters()
                            if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name)]}
            ]

        #1
        elif args.prune_fc:
            params = [
                {'params': [param for name, param in model.named_parameters()
                            if not ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name
                                    or '_fc.weight' in name)],
                                    'weight_decay': args.weight_decay},
                {'params': [param for name, param in model.named_parameters()
                            if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name
                                or '_fc.weight' in name)]}
            ]
        #2
        elif args.prune_se:
            params = [
                {'params': [param for name, param in model.named_parameters()
                            if not ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name
                                    or '_fc.weight' in name
                                    or '_se_expand.weight' in name or '_se_reduce.weight' in name)],
                                    'weight_decay': args.weight_decay},
                {'params': [param for name, param in model.named_parameters()
                            if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name
                                    or '_fc.weight' in name
                                    or '_se_expand.weight' in name or '_se_reduce.weight' in name)]}
            ]
        #3
        elif args.prune_dw:
            params = [
                {'params': [param for name, param in model.named_parameters()
                            if not ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name
                                    or '_fc.weight' in name
                                    or '_se_expand.weight' in name or '_se_reduce.weight' in name
                                    or '_depthwise_conv' in name)],
                                    'weight_decay': args.weight_decay},
                {'params': [param for name, param in model.named_parameters()
                            if ('_project_conv' in name or '_expand_conv' in name or '_conv_head' in name
                                    or '_fc.weight' in name
                                    or '_se_expand.weight' in name or '_se_reduce.weight' in name
                                    or '_depthwise_conv' in name)]}
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

    # If multipile GPUs are used
    if use_cuda and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # Transfers model to device (GPU/CPU).
    model.to(device)

    # START TRAINING

    # Initializing training variables
    best_acc = 0
    all_losses = []
    start_time = time.time()
    if args.slurm_save == None:
        logfile = open('./frozen_assignment_retraining/saved_models/logfile.txt', 'a+')
    else:
        logfile = open(args.slurm_save + '/logfile.txt', 'a+')

    model.train()

    for epoch in range(args.epochs):

        # Executing training process
        running_loss, running_accuracy = train(train_loader, model, loss_fct, optimizer, epoch)

        # Evaluation
        model.eval()
        val_loss, val_accuracy = evaluate(model, loss_fct, val_loader)

        # Logging the accuracies
        all_losses += [(epoch, running_loss, val_loss, running_accuracy, val_accuracy)]
        print('Epoch {0} running loss {1:.3f} val loss {2:.3f}  running acc {3:.3f} '
              'val acc{4:.3f}  time {5:.3f}'.format(*all_losses[-1], time.time() - start_time))
        logfile.write('Epoch {0} running loss {1:.3f} val loss {2:.3f}  running acc {3:.3f} '
                      'val acc{4:.3f}  time {5:.3f}\n'.format(*all_losses[-1], time.time() - start_time))

        if args.slurm_save == None:
            cp_location = './frozen_assignment_retraining/saved_models/checkpoint.pt'
            sd_location = './frozen_assignment_retraining/saved_models/best_model.pt'
        else:
            cp_location = args.slurm_save + '/checkpoint.pt'
            sd_location = args.slurm_save + '/best_model.pt'

        # Saving checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': val_accuracy,
            'lr': optimizer.param_groups[0]['lr']
        }, cp_location)


        # Checking if current epoch yielded best validation accuracy
        is_best = val_accuracy > best_acc
        best_acc = max(val_accuracy, best_acc)

        # If so, saving best model state_dict
        if is_best and epoch > 0:
            torch.save(model.state_dict(), sd_location)

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

        else:
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

    train_w_frozen_assignment(train_loader, val_loader, model, loss_fct)


if __name__ == '__main__':
    t1_start = time.perf_counter()
    main()
    t1_stop = time.perf_counter()
    print("Elapsed time: %.2f [s]" % (t1_stop-t1_start))
