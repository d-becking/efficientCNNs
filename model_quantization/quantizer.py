"""
This document contains the engine for entropy controlled ternary quantization. It receives all hyperparameters
from the "main_trained_quantization" document.

References:
    https://github.com/TropComplique/trained-ternary-quantization
    MIT License - Copyright (c) 2017 Dan Antoshchenko
    https://github.com/uoguelph-mlrg/Cutout
    Educational Community License, Version 2.0 (ECL-2.0) - Copyright (c) 2019 Vithursan Thangarasa
    https://github.com/lukemelas/EfficientNet-PyTorch
    Apache License, Version 2.0 - Copyright (c) 2019 Luke Melas-Kyriazi
    https://github.com/akamaster/pytorch_resnet_cifar10
    Yerlan Idelbayev's ResNet implementation for CIFAR10/CIFAR100 in PyTorch


This document contains the following functions/classes:

    - train_quantization(model, loss_fct, train_iterator, val_iterator, optimizer, opt_fp, opt_sf, lambda_max_divider,
                            initial_assignment, initial_c_scaler, plot_flag, n_epochs, stop_quantizing_flag=0)

        - optimization_step(model, loss, data, labels, optimizer, opt_fp, opt_sf,
                                lambda_max_divider, assignment_list, lambda_list, stop_quantizing_flag=0)

            - get_grads(t_grad, fp_weights, w_p, w_n, assignment)
            - quantize(weights, w_p, w_n, lambda_max_divider, Lambda, lambda_decay, cuda)

                - get_distances(weights, w_p, w_n)
                - apply_entropy_constraint(weights, w_p, w_n, lambda_decay, lambda_max_divider)

                    - get_pmf(prelim_assignment)

    - give_net_stats(optim_param_group, quant_weights_value_log, quant_weights_counts_log, entropy, ini_flag=False)
    - calc_sparsity(optimizer, total_params, total_quant_params)
    - evaluate(model, loss, val_iterator)
    - plot_weights(nr_layer, counts, values, entropy, acc, sparsity, lambda_max_div, initial_c_scaler,
                        plot_flag, stop_quantizing_flag, nr_of_t_layers)
"""

# Imports
import numpy as np
import torch
import time
#import matplotlib.pyplot as plt
import os
#from torch.utils.tensorboard import SummaryWriter


def get_pmf(prelim_assignment):
    """
    Calculates the probabilitiy mass function (pmf) which here is simplified the number of weights assigned to a
    specific centroid devided by th number of all weights. The preliminary assignment considers only the minimal
    distance from  centroids to weights as cost.
    With "spars_bound" we ensure that at least 50% of all weights would be assigned to the zero-centroid w_0 such that
    the entropy score (information content) for w_0 is always the lowest.

    Parameters:
    -----------
        prelim_assignment:
            Minimal arguments for all 3 centroid distances to all layer weights

    Returns:
    --------
        pmf_prelim:
            Percentage frequencies of -only distance dependent- centroid assignments (w_n, w_0, w_p)
            with pmf[w_n] + pmf[w_0] + pmf[w_p] = 1 and  pmf[w_0] always > 0.5
    """
    C_counts = torch.ones(3)
    C_val, C_cts = torch.unique(prelim_assignment, return_counts=True)

    # For the usual case that layer weights are assigned to three centroids (ternary)
    if C_cts.shape[0] == 3:
        C_counts = C_cts

    # The following two cases, especially the last one, should not occur as precautions were taken such that the
    # w_0 centroid can't absorb all assignments
    # If layer weights are assigned to two only centroids (binary)
    elif C_cts.shape[0] == 2:
        if 0 not in C_val:
            C_counts[1] = C_cts[0]
            C_counts[2] = C_cts[1]
        if 1 not in C_val:
            C_counts[0] = C_cts[0]
            C_counts[2] = C_cts[1]
        if 2 not in C_val:
            C_counts[0] = C_cts[0]
            C_counts[1] = C_cts[1]
    # If layer weights are assigned to only one centroid
    elif C_cts.shape[0] == 1:
        if (0 not in C_val and 1 not in C_val):
            C_counts[2] = C_cts[0]
        if (0 not in C_val and 2 not in C_val):
            C_counts[1] = C_cts[0]
        if (1 not in C_val and 2 not in C_val):
            C_counts[0] = C_cts[0]

    pmf_prelim = torch.div(C_counts.type(torch.float32), torch.numel(prelim_assignment))

    # Ensuring that at least 50% of all weights are assigned to w_0 and probabilities still sum up to 1
    spars_bound = 0.5
    if pmf_prelim[1] < spars_bound:
        pmf_prelim[0] -= (pmf_prelim[0]/(pmf_prelim[0] +
                                        pmf_prelim[2]))*(spars_bound - pmf_prelim[1])
        pmf_prelim[2] -= (pmf_prelim[2] / (pmf_prelim[0] +
                                           pmf_prelim[2])) * (spars_bound - pmf_prelim[1])
        pmf_prelim[1] = spars_bound

    return pmf_prelim


def get_distances(weights, w_p, w_n):
    """
    Calculates the squared distances from all layer weights to all centroids (w_n, w_0, w_p).

    Parameters:
    -----------
        weights:
            Full precision weights of the given layer.
        w_p, w_n:
            Negative and positive centroid values of a layer.

    Returns:
    --------
        squared distances to all 3 centroids in a new tensor axis
    """
    if use_cuda:
        C = torch.tensor([-w_n, 0, w_p]).cuda()
    else:
        C = torch.tensor([-w_n, 0, w_p])

    # Adding new axes to the centroid vector C to make the subtraction/distance calculation with the layer's
    # weights shape possible
    G = torch.unsqueeze(C, 1)
    for i in range(1, weights.shape.__len__()):
        # Iteratively adding a new axis for the layers shape's length
        G = torch.unsqueeze(G, 1)

    return (G.sub(weights))**2


def apply_entropy_constraint(weights, w_p, w_n, lambda_decay, lambda_max_divider):
    """
    Applies the entropy constraint to the quantizer's cost function for a given lambda_max_divider. The function
    finds the maximum Lambda (for which almost all weights are assigned to w_0) with a top-down-middle-up-procedure
    and scales it with lambda_decay and lambda_max_divider. The returned cost can be described as:

        d(w_i, w_c) + (lambda_max_divider * lambda_decay * Lambda * information_content_c)

            c ... element of the centroids [w_n, w_0, w_p],
            i ... element of i layer weights and
            d() ... the squared distance of layer weights to the centroids.

    Parameters:
    -----------
        weights:
            Full precision weights of the given layer.
        w_p, w_n:
            Negative and positive centroid values of a layer.
        lambda_decay:
            Decreases Lambda according to the number of layer weights. Smaller layers won't be as sparse as huge layers.
        lambda_max_divider:
            Parameter on which grid search is applied. Lambda_max is defined as the Lambda for which almost all layer
            weights would be assigned to w_0. The greater lambda_max_divider the sparser will be the resulting ternary
            network. We want to find the maximal lambda_max_divider, i.e. the sparsest network, which still maintains
            the initial accuracy (to a specified level).

    Returns:
    --------
        cost:
            Final cost for all centroids with lambda_max_divider and lambda_decay applied.
        Lambda:
            Intensity of entropy constraint.
    """
    # Calculating distances from layer weights to centroids
    dist = get_distances(weights, w_p, w_n)
    # Preliminary assignment depending only on the distance as cost
    prelim_assignment = torch.argmin(dist, dim=0)
    # Calculating probability mass function approximation, given the preliminary assignment
    pmf = get_pmf(prelim_assignment)

    # Centroid's information content
    if use_cuda:
        I = -torch.log2(pmf).cuda()
    else:
        I = -torch.log2(pmf)

    # Extrude information content I such that it has the same num of dimensions as "weights"
    I_extruded = torch.unsqueeze(I, 1)
    for i in range(1, dist.shape.__len__() - 1):
        # Iteratively adding a new axis for the layers shape's length
        I_extruded = torch.unsqueeze(I_extruded, 1)

    # Calculate the maximal Lambdas for which all weights would be assigned to either [w_0, w_n] or [w_0, w_p]
    Lambda_n_max = ((0 - weights.min().abs())**2 - (w_n.sub(weights.min().abs()))**2) / (I[0] - I[1])
    Lambda_p_max = ((0 - weights.max().abs())**2 - (w_p.sub(weights.max().abs()))**2) / (I[2] - I[1])

    # From the maximal Lambdas choose the smaller value to prevent that layer weights become binary or even unary
    if Lambda_p_max < Lambda_n_max:
        Lambda = Lambda_p_max
    else:
        Lambda = Lambda_n_max

    # Multiplying Lambda with its scaling factors
    Lambda *= lambda_max_divider
    Lambda *= lambda_decay

    # Final cost
    cost = torch.add(dist, Lambda * I_extruded)

    return cost, Lambda

def quantize(weights, w_p, w_n, lambda_max_divider, Lambda, lambda_decay, cuda, pruning=False):
    """
    Quantizes given weights in two modi: Lambda==0 (i.e. Nearest Neighbor quantization), or Lambda!=0 (i.e. an
    entropy constraint is applied).

    Parameters:
    -----------
        weights:
            Full precision weights.
        w_p, w_n:
            Negative (i.e. < 0) and positive (i.e. > 0) valued centroids of a layer.
        lambda_max_divider:
            Scalar by which the maximum lambda (Lambda is lambda_max if almost all weights would be assigned to the most
            probable centroid, which is the zero-centroid w_0) is multiplied.
        Lambda:
            Intensity of entropy constraint. Initial Lambda will be updated every step and be feed back to this fct.
        lambda_decay:
            Scaling factor for Lambda values depending on the number of parameters per layer (i.e. greater layer ->
            greater Lambda).
        cuda:
            Global parameter received from main_trained_quantization.py. If True: training on GPU.

    Returns:
    --------
        Quantized weights of a layer.
        lambda_return:
            updated Lambda per layer.
        assignment:
            the assignment of the ternary net weights to the updated centroids.
    """

    # Making use_cuda variable global such that all functions in quantizer.py can use it
    global use_cuda, prune
    use_cuda = cuda
    prune = pruning

    # No entropy constraint, only KNN
    if Lambda == 0:
        cost = get_distances(weights, w_p, w_n)
        lambda_return = 0.0

    # Entropy constraint
    if Lambda != 0:
        cost, lambda_return = apply_entropy_constraint(weights, w_p, w_n, lambda_decay, lambda_max_divider)

    # Assigning weights to centroids with minimal cost
    assignment = torch.argmin(cost, dim=0)

    a = (assignment == 2).float()
    b = (assignment == 0).float()

    quant_weights = w_p*a + (-w_n*b)

    if prune:
        quant_weights = (a + b) * weights

    ###################  TTQ paper ######################
    # max_w = weights.abs().max()
    # norm_weights = weights / max_w
    # delta = 0.05
    # a = (norm_weights > delta).float()S
    # b = (norm_weights < -delta).float()
    # c = torch.ones(weights.size()).cuda() - a - b
    # assignment = (2 * a) + c
    # lambda_return = 0.0
    #####################################################

    return quant_weights, lambda_return, assignment


def get_grads(t_grad, fp_weights, w_p, w_n, assignment):
    """
    Returns resulting gradients for the centroids and for the full precision copy of the model.

    Parameters:
    -----------
        t_grad:
            Gradients of the (ternary) quantized layer
        fp_weights:
            Corresponding full precision weights
        w_p, w_n:
            Negative (i.e. < 0) and positive (i.e. > 0) valued centroids of a layer
        assignment:
            Assignment matrix of the given layer's shape containing the assigned centroids

    Returns:
    --------
        Gradient for the full precision weights
        Gradient for w_p
        Gradient for w_n
    """

    a = (assignment == 2).float()
    b = (assignment == 0).float()

    if use_cuda:
        c = torch.ones(fp_weights.size()).cuda() - a - b
    else:
        c = torch.ones(fp_weights.size()) - a - b

    return w_p*a*t_grad + w_n*b*t_grad + 1.0*c*t_grad, (a*t_grad).sum(), (b*t_grad).sum()


def optimization_step(model, loss, data, labels, optimizer, opt_fp, opt_sf,
                      lambda_max_divider, assignment_list, lambda_list, stop_quantizing_flag=0):
    """
    Make forward and backward pass, get model gradients, update weights and centroids and requantize ternary layers

    Parameters:
    -----------
        model:
            The neural network model.
        loss:
            Loss function to use (e.g. cross entropy, MSE, ...).
        data:
            Training data from PyTorch Dataloader of given batch size.
        labels:
            Corresponding labels/targets to training data.
        optimizer:
            Adam optimizer with two parameter groups: [0] for layers which will not be quantized (BatchNorm, Bias, ...)
            and [1] with layers that will be quantized ternary. Learning rate 1.
        opt_fp:
            Adam optimizer for the full precision copy of the initial model.
        opt_sf:
            Adam optimizer for the centroids (w_n, w_p) per layer.
        lambda_max_divider:
            Parameter on which grid search is applied. Lambda_max is defined as the Lambda for which almost all layer
            weights would be assigned to w_0. The greater lambda_max_divider the sparser will be the resulting ternary
            network. We want to find the maximal lambda_max_divider, i.e. the sparsest network, which still maintains
            the initial accuracy (to a specified level).
        assignment_list:
            Assignment matrices per layer containing the assigned centroids.
        lambda_list:
            Updated and scaled Lambda per layer.
        stop_quantizing_flag:
            If set to 1 the centroid's values (w_n, w_p) will be further optimized but without a quantization step
            such that the spatial assignment of centroids within a layer is frozen

    Returns:
    --------
        batch_loss:
            Loss value of given batch.
        batch_accuracy:
            Prediction accuracy for given batch.
        lambda_list:
            Updated lambda_list (only if stop_quantizing_flag is not set).
        assignment_list:
            Updated assignment_list (only if stop_quantizing_flag is not set).
    """

    # Resetting variables for calculating current batch accuracy
    correct = 0
    total = 0

    # Data and targets to GPU
    if use_cuda:
        data, labels = data.cuda(), labels.cuda(non_blocking=True)

    # Forward pass using quantized model
    outputs = model(data)

    # Compute loss
    batch_loss = loss(outputs, labels)

    # Compute accuracy
    _, predicted = outputs.max(1)
    # delete outputs in order to prevent RAM overflow
    del outputs
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
    batch_accuracy = 100. * correct / total

    # Setting optimizers' gradients to zero
    optimizer.zero_grad()
    opt_fp.zero_grad()
    opt_sf.zero_grad()

    # Compute gradients for quantized model, backward path
    batch_loss.backward()

    # Get all quantized weights
    ternary_weights = optimizer.param_groups[1]['params']

    if prune:
        pretrained_ternary = optimizer.param_groups[2]['params']

    # Get their full precision backups
    fp_weights = opt_fp.param_groups[0]['params']

    # Get centroids for each quantized layer
    centroids = opt_sf.param_groups[0]['params']

    # Calculating total number of ternary weights
    total_num_weights = 0
    for w in ternary_weights:
        total_num_weights += torch.numel(w)

    # Number or weights in the largest layer (needed for lambda_decay)
    num_layer_weights = []
    for l in range(len(ternary_weights)):
        num_layer_weights.append(ternary_weights[l].numel())
    largest_layer = max(num_layer_weights)

    # Getting full precision and centroids' gradients layerwise
    for i in range(len(ternary_weights)):

        # Get centroids per layer
        w_p, w_n = centroids[i].data[0], centroids[i].data[1]

        # Get modified gradients
        fp_grad, w_p_grad, w_n_grad = get_grads(ternary_weights[i].grad.data,
                                                fp_weights[i].data, w_p, w_n, assignment_list[i])

        # Gradient for full precision weights
        fp_weights[i].grad = fp_grad

        # Gradient for centroids
        if use_cuda:
            centroids[i].grad = torch.tensor([w_p_grad, w_n_grad]).cuda()
        else:
            centroids[i].grad = torch.tensor([w_p_grad, w_n_grad])

        # Setting gradients of ternary weights to zero
        ternary_weights[i].grad.data.zero_()

    # Setting gradients of pretrained ternary weights to zero to remain layer architecture
    if prune:
        for i in range(len(pretrained_ternary)):
            pretrained_ternary[i].grad.data.zero_()

    # Update non quantized weights (bias and batch norm layers)
    optimizer.step()

    # Update all full precision weights if stop_quantizing_flag is not set
    if not stop_quantizing_flag:
        opt_fp.step()

    # Update all centroids with (Adam) optimizer
    opt_sf.step()

    # Update all quantized weights with updated full precision weights
    for j in range(len(ternary_weights)):

        # Get centroids per layer
        w_p, w_n = centroids[j].data[0], centroids[j].data[1]

        # To prevent that w_n or w_p become very small or equal to 0 which would result in a binary/unary layer
        std05 = torch.std(fp_weights[j].data).mul(0.675)
        if centroids[j].data[0] < std05:
            centroids[j].data[0] = std05
        if centroids[j].data[1] < std05:
            centroids[j].data[1] = std05

        # If stop_quantizing_flag is set: freeze assignment, update centroid values only
        if stop_quantizing_flag:
            ternary_weights[j].data = w_p * (assignment_list[j] == 2).float() + (
                                        -w_n * (assignment_list[j] == 0).float())

        # Full update and quantization step including weight assignment
        else:
            # Requantize ternary weights layerwise using updated weights and centroids
            lambda_decay = ternary_weights[j].numel() / largest_layer
            # To ensure that the decay is not too intense we introduce a sustain
            lambda_sustain = 0.05
            lambda_decay = (lambda_decay + lambda_sustain) / (1 + lambda_sustain)
            ternary_weights[j].data, lambda_return, assignment = quantize(fp_weights[j].data, w_p, w_n,
                                                                          lambda_max_divider, lambda_list[j],
                                                                          lambda_decay, use_cuda, pruning=prune)
            lambda_list[j] = lambda_return
            assignment_list[j] = assignment

    return batch_loss, batch_accuracy, lambda_list, assignment_list


def give_net_stats(optim_param_group, quant_weights_value_log, quant_weights_counts_log, entropy, ini_flag=False):
    """
    Returns network statistics regarding to the centroids distributions and resulting entropy per layer.

    Parameters:
    -----------
        optim_param_group:
            An optimizer's parameter group which should be analyzed. Here we're interested in the ternary weights which
            can be found in "optimizer.param_groups[1]['params']".
        quant_weights_value_log:
            Logger for the centroids' values over all epochs.
        quant_weights_counts_log:
            Logger for the number of weights assigned to the centroids over all epochs.
        entropy:
            Logger for layerwise entropy over all epochs.
        ini_flag:
            Set True for initial calculation of network stats. Set False afterwards to concatenate updated stats.

    Returns:
    --------
        quant_weights_value_log:
            History of centroid values plus the latest update concatenated layerwise (if ini_flag==False).
        quant_weights_counts_log:
            History of weights per centroid plus the latest update concatenated layerwise (if ini_flag==False).
        entropy:
            History of entropy plus the latest update concatenated layerwise (if ini_flag==False).
    """

    for i, layer in enumerate(optim_param_group):

        C_counts = torch.zeros(3)
        C_values = torch.zeros(3)

        values, counts = torch.unique(layer, return_counts=True)

        # For the usual case that layer weights are assigned to three centroids (ternary)
        if counts.shape[0] == 3:
            C_counts = counts
            C_values = values

        # The following cases, especially the last one, should not occur as precautions were taken such that the w_0
        # centroid can't absorb all other assignments
        # If layer weights are assigned to two only centroids (binary)
        elif counts.shape[0] == 2:
            if values[0] == 0:  # 0 and W_p available
                C_counts[1] = counts[0]
                C_counts[2] = counts[1]
                C_values[2] = values[1]
            if values[1] == 0:  # W_n and 0 available
                C_counts[0] = counts[0]
                C_counts[1] = counts[1]
                C_values[0] = values[0]
            if 0 not in values:  # W_n and W_p available
                C_counts[0] = counts[0]
                C_counts[2] = counts[1]
                C_values[0] = values[0]
                C_values[2] = values[1]
        # If layer weights are assigned to only one centroid
        elif counts.shape[0] == 1:
            if values[0] == 0:
                C_counts[1] = counts[0]
            if values[0] < 0:
                C_counts[0] = counts[0]
            if values[0] > 0:
                C_counts[2] = counts[0]

        # Initialization of the logger list
        if ini_flag:
            quant_weights_value_log += [C_values.unsqueeze(0)]
            quant_weights_counts_log += [C_counts.unsqueeze(0)]
            pmf = get_pmf(layer)
            entropy += [-torch.sum(torch.mul(pmf, torch.log2(pmf))).unsqueeze(0)]
        # List concatenating with new stats
        else:
            pmf = get_pmf(layer)
            H = -torch.sum(torch.mul(pmf, torch.log2(pmf)))
            quant_weights_value_log[i] = torch.cat((quant_weights_value_log[i],
                                                    C_values.type_as(quant_weights_value_log[i]).unsqueeze(0)), dim=0)
            quant_weights_counts_log[i] = torch.cat((quant_weights_counts_log[i],
                                                     C_counts.type_as(quant_weights_counts_log[i]).unsqueeze(0)), dim=0)
            entropy[i] = torch.cat((entropy[i], H.type_as(entropy[i]).unsqueeze(0)), dim=0)

    return quant_weights_value_log, quant_weights_counts_log, entropy


def calc_sparsity(optimizer, total_params, total_quant_params):
    """
    Returns the sparsity of the overall network and the sparsity of quantized layers only.

    Parameters:
    -----------
        optimizer:
            An optimizer containing quantized model layers in param_groups[1]['params'] and non-quantized layers,
            such as BatchNorm, Bias, etc., in param_groups[1]['params'].
        total_params:
            Total number of parameters.
        total_quant_params:
            Number of quantized parameters.

    Returns:
    --------
        sparsity_total:
            Sparsity of the overall network.
        sparsity_quant:
            Sparsity of quantized layers of the network.
    """

    nonzero_elements_quant = 0
    for layer in optimizer.param_groups[1]['params']:
        nonzero_elements_quant += layer[layer != 0].numel()

    nonzero_elements_total = 0
    for layer in optimizer.param_groups[0]['params']:
        nonzero_elements_total += layer[layer != 0].numel()
    nonzero_elements_total += nonzero_elements_quant

    sparsity_total = (total_params - nonzero_elements_total) / total_params
    sparsity_quant = (total_quant_params - nonzero_elements_quant) / total_quant_params

    return sparsity_total, sparsity_quant


def evaluate(model, loss, val_iterator):
    """
    Evaluating the model (no backward path / optimization step)

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

        for data, labels in val_iterator:

            correct = 0
            total = 0

            if use_cuda:
                data = data.cuda()
                labels = labels.cuda(non_blocking=True)

            n_batch_samples = labels.size()[0]
            logits = model(data)

            # Compute batch loss
            batch_loss = loss(logits, labels)

            # Compute batch accuracy
            _, predicted = logits.max(1)
            # Delete logits to prevent RAM overflow
            del logits
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            batch_accuracy = 100. * correct / total

            loss_value += batch_loss.float()*n_batch_samples
            # Delete batch_loss to prevent RAM overflow
            del batch_loss
            accuracy += batch_accuracy*n_batch_samples
            total_samples += n_batch_samples

        return loss_value/total_samples, accuracy/total_samples


def plot_weights(nr_layer, counts, values, entropy, acc, sparsity, lambda_max_div, initial_c_scaler,
                 plot_flag, stop_quantizing_flag, nr_of_t_layers):
    """
    Plots the model's ternarization process and fine-tuning in terms of centroid distribution, sparsity, accuracy and
    entropy over epochs for each layer. Saving the plot data as numpy files possible, therefore change save_as_np=False.

    Parameters:
    -----------
        nr_layer:
            Iterates over all quantized layers of the network.
        counts:
            Number of assigned weights per centroid [w_n, w_0, w_p].
        values:
            Centroid values (cluster centers).
        entropy:
            Layer's entropy -sum_i(P_i*log2(P_i)), i is element [w_n, w_0, w_p].
        acc:
            Overall accuracy history.
        sparsity:
            Overall total network sparsity.
        lambda_max_div:
            Parameter on which grid search is applied. Lambda_max is defined as the Lambda for which almost all layer
            weights would be assigned to w_0. We want to find the maximal lambda_max_divider, i.e. the sparsest network,
            which still maintains the initial accuracy (to a specified level).
        initial_c_scaler:
            For the first optimization step all centroids (i.e. cluster centers [w_n, 0, w_p] in the ternary case)
            will get the values w_n  = w_min * initial_c_scaler and respectively w_p = w_max * initial_c_scaler.
        plot_flag:
            If True the ternarization process over epochs will be plotted.
        stop_quantizing_flag:
            If set to 1 the centroid's values (w_n, w_p) will be further optimized but without a quantization step
            such that the spatial assignment of centroids within a layer is frozen.
        nr_of_t_layers:
            Number of quantized / "ternary" layers.
    """

    # X-Axis
    x = np.arange(0, len(counts[nr_layer]))
    # Number of assigned weights per centroid
    y1 = np.transpose(counts[nr_layer].cpu().detach().numpy())
    # Negative centroid values w_n
    y2 = values[nr_layer].cpu().detach().numpy()[:, 0]
    # Positive centroid values w_p
    y3 = values[nr_layer].cpu().detach().numpy()[:, 2]
    # Zeros for w_0
    y4 = np.zeros(len(values[nr_layer]))
    # Layer's entropy
    y5 = entropy[nr_layer].cpu().detach().numpy()
    # Overall accuracy
    y6 = acc
    # Overall sparsity
    y7 = sparsity

    save_as_np = False
    if save_as_np:
        np.save('./model_quantization/logfiles/plots/x_axis_layer'+ str(nr_layer)+'_lambda_max_div'+str(lambda_max_div)
                                                        + '_ini_c_scaler' + str(initial_c_scaler)
                                                        + '_stopquant'+ str(stop_quantizing_flag)+'.npy', x)
        np.save('./model_quantization/logfiles/plots/centroid_counts_log_layer'+ str(nr_layer)+'_lambda_max_div'
                                                        + str(lambda_max_div) + '_ini_c_scaler' + str(initial_c_scaler)
                                                        + '_stopquant'+ str(stop_quantizing_flag)
                                                        + '.npy', y1)
        np.save('./model_quantization/logfiles/plots/w_n_log_layer'+ str(nr_layer)+'_lambda_max_div'+str(lambda_max_div)
                                                        + '_ini_c_scaler' + str(initial_c_scaler)
                                                        + '_stopquant'+ str(stop_quantizing_flag)+'.npy', y2)
        np.save('./model_quantization/logfiles/plots/w_p_log_layer'+ str(nr_layer)+'_lambda_max_div'+str(lambda_max_div)
                                                        + '_ini_c_scaler' + str(initial_c_scaler)
                                                        + '_stopquant'+ str(stop_quantizing_flag)+'.npy', y3)
        np.save('./model_quantization/logfiles/plots/w_0_log_layer'+ str(nr_layer)+'_lambda_max_div'+str(lambda_max_div)
                                                        + '_ini_c_scaler' + str(initial_c_scaler)
                                                        + '_stopquant'+ str(stop_quantizing_flag)+'.npy', y4)
        np.save('./model_quantization/logfiles/plots/entropy_log_layer'+ str(nr_layer)+'_lambda_max_div'
                                                        + '_ini_c_scaler' + str(initial_c_scaler) + str(lambda_max_div)
                                                        + '_stopquant'+ str(stop_quantizing_flag) + '.npy', y5)
        np.save('./model_quantization/logfiles/plots/acc_log_layer'+ str(nr_layer)+'_lambda_max_div'+str(lambda_max_div)
                                                        + '_ini_c_scaler' + str(initial_c_scaler)
                                                        + '_stopquant'+ str(stop_quantizing_flag)+'.npy', y6)
        np.save('./model_quantization/logfiles/plots/sparsity_log_layer'+ str(nr_layer)+'lambda_max_div'
                                                        + str(lambda_max_div) + '_ini_c_scaler' + str(initial_c_scaler)
                                                        + '_stopquant'+ str(stop_quantizing_flag) + '.npy', y7)

    if plot_flag:
        plt.figure(nr_layer+(stop_quantizing_flag*nr_of_t_layers))

        plt.subplot(311)
        plt.stackplot(x, y1, labels=['W_n', '0', 'W_p'])
        plt.legend(loc='upper left')
        plt.title('layer {}, lambda_max_div {}'.format(nr_layer, lambda_max_div))

        plt.subplot(312)
        plt.plot(x, y2, 'b', x, y3, 'g', x, y4, 'k')

        plt.subplot(313)
        plt.plot(x, y5, 'b', label='entropy')
        plt.legend(loc='upper left')
        ax2 = plt.twinx()
        ax2.plot(x, y6, 'r', label='test accuracy')
        ax2.plot(x, y7, 'g', label='sparsity')
        ax2.legend(loc='upper right')

        plt.savefig('./model_quantization/logfiles/plots/plot_layer' + str(nr_layer)
                    + '_lambda_max_div' + str(lambda_max_div) + '_ini_c_scaler' + str(initial_c_scaler) + '_stopquant'
                    + str(stop_quantizing_flag) + '.pdf')
        plt.close('all')

def train_quantization(model, loss_fct, train_iterator, val_iterator, optimizer, opt_fp, opt_sf,
                       lambda_max_divider, initial_assignment, initial_c_scaler, resume, cuda, pruning, plot_flag,
                       n_epochs, slurm_save, stop_quantizing_flag=0):
    """
    Main loop of this document ('quantizer.py'). Parameters received from 'main_trained_quantization.py', the main
    document of the model_quantization package. Here the quantization optimization_step is executed iteratively.

    Parameters:
    -----------
        model:
            The neural network model.
        loss_fct:
            Loss function to use (e.g. cross entropy, MSE, ...).
        train_iterator:
            PyTorch Dataloader for given train dataset.
        val_iterator:
            PyTorch Dataloader for given validation dataset.
        optimizer:
            Adam optimizer with two parameter groups: [0] for layers which will not be quantized (BatchNorm, Bias, ...)
            and [1] with layers that will be quantized ternary. Learning rate 1.
        opt_fp:
            Adam optimizer for updating the full precision copy of the model.
        opt_sf:
            Adam optimizer for the centroids (w_n, w_p) per layer.
            Learning rate 2 < learning rate 1 < approx. 0.1 or 0.01 * learning rate of pretraining
        lambda_max_divider:
            Parameter on which grid search is applied. Lambda_max is defined as the Lambda for which almost all layer
            weights would be assigned to w_0. The greater lambda_max_divider the sparser will be the resulting ternary
            network. We want to find the maximal lambda_max_divider, i.e. the sparsest network, which still maintains
            the initial accuracy (to a specified level).
        initial_assignment:
            Initial cluster center values before they're trained (dependent on initial_c_scaler).
        initial_c_scaler:
            For the first optimization step all centroids (i.e. cluster centers [w_n, 0, w_p] in the ternary case)
            will get the values w_n  = w_min * initial_c_scaler and respectively w_p = w_max * initial_c_scaler.
        resume:
            Path to checkpoint. If it exists, resume.
        cuda:
            Global parameter received from main_trained_quantization.py. If True: training on GPU.
        plot_flag:
            If True the ternarization process over epochs will be plotted.
        n_epochs:
            For how many epochs the optimization step should be executed.
        stop_quantizing_flag:
            If set to 1 the centroid's values (w_n, w_p) will be further optimized but without a quantization step
            such that the spatial assignment of centroids within a layer is frozen.
    """

    # Making use_cuda variable global such that all functions in quantizer.py can use it
    global use_cuda, prune
    use_cuda = cuda
    prune = pruning

    #tb = SummaryWriter("tb")

    # Initializing variables
    start_time = time.time()
    all_losses = []
    running_loss = 0.0
    running_accuracy = 0.0
    best_acc = 0
    n_steps = 0
    val_acc = [1.0]
    sparsity_log = [0.0]
    nr_epochs = 0
    lambda_list = []
    total_params = 0
    total_quant_params = 0

    # Initialize lambda_list (one Lambda per ternary layer) and counting number of quantized parameters
    for layer in optimizer.param_groups[1]['params']:
        lambda_list += [1]
        total_quant_params += layer.numel()

    # Calculating total number of parameters
    for layer in optimizer.param_groups[0]['params']:
        total_params += layer.numel()
    total_params += total_quant_params

    # Initialize the assignment of full precision weights to initial cluster centers (centroids)
    assignment_list = initial_assignment

    # Resume variables from checkpoint if it exists
    if resume and os.path.isfile(resume):
        checkpoint = torch.load(resume)
        best_acc = checkpoint['best_acc']
        lambda_list = checkpoint['lambda_list']
        start_epoch = checkpoint['epoch']
        # When ternary value training with frozen assignment subsequently follows the full quantization process
        if stop_quantizing_flag != checkpoint['stop_quantizing_flag']:
            start_epoch = 0
    else:
        start_epoch = 0

    # Initial calculation of network statistics
    t_weights_value_log, t_weights_counts_log, entropy = give_net_stats(optimizer.param_groups[1]['params'],
                                                                        [], [], [], ini_flag=True)
    sparsity_total, sparsity_quant = calc_sparsity(optimizer, total_params, total_quant_params)
    print('sparsity total {:.2f}%'.format(sparsity_total * 100))
    print('sparsity of quantized layers {:.2f}%'.format(sparsity_quant * 100))

    # START TRAINING
    model.train()

    for epoch in range(start_epoch, n_epochs):

        timer = 0

        for data, labels in train_iterator:

            end = time.time()

            # Main training step
            batch_loss, batch_accuracy, lambda_list, assignment_list = optimization_step(
                                                model, loss_fct, data, labels, optimizer, opt_fp, opt_sf,
                                                lambda_max_divider, assignment_list, lambda_list, stop_quantizing_flag)

            # Summing up batch losses and accuracies over each step
            running_loss += batch_loss.float()
            # Delete batch_loss to prevent RAM overflow
            del batch_loss
            running_accuracy += batch_accuracy
            n_steps += 1

            timer += time.time() - end
            # Printing preliminary results
            if n_steps % 100 == 0:
                total_steps = int(len(train_iterator.dataset) / train_iterator.batch_size) + 1
                print('Epoch: {}, step: {}/{}, running_loss: {:.4f}, batch_acc: {:.4f}, running_acc: {:.4f}, '
                      'elapsed time: {:.2f} s'.format(epoch, n_steps, total_steps, running_loss/n_steps,
                                                batch_accuracy, running_accuracy/n_steps, timer))
                timer = 0

        if slurm_save == None:
            checkpoint_path = './model_quantization/saved_models/checkpoint.pt'
            ternary_best_path = './model_quantization/saved_models/Ternary_best_acc.pt'
            assignments_path = './model_quantization/saved_models/assignments.pt'
            centroids_path = './model_quantization/saved_models/centroids.pt'
        else:
            checkpoint_path = slurm_save + '/checkpoint.pt'
            ternary_best_path = slurm_save + '/Ternary_best_acc.pt'
            assignments_path = slurm_save + '/assignments.pt'
            centroids_path = slurm_save + '/centroids.pt'

        # Saving checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict() if not torch.cuda.device_count() > 1
                                                else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'opt_fp_state_dict': opt_fp.state_dict(),
            'opt_sf_state_dict': opt_sf.state_dict(),
            'best_acc': best_acc,
            'centroids': opt_sf.param_groups[0]['params'],
            'fp_update': opt_fp.param_groups[0]['params'],
            'assignment_list': assignment_list,
            'lambda_list': lambda_list,
            'stop_quantizing_flag': stop_quantizing_flag
        }, checkpoint_path)

        # Evaluation
        val_loss, val_accuracy = evaluate(model, loss_fct, val_iterator)

        # Checking if current epoch yielded best validation accuracy
        is_best = val_accuracy > best_acc
        best_acc = max(val_accuracy, best_acc)

        # If so, saving best model state_dict, cluster assigments and cluster centers. Update the parameter text file.
        if epoch > 0 and is_best:
            if not torch.cuda.device_count() > 1:
                torch.save(model.state_dict(), ternary_best_path)
            else:
                torch.save(model.module.state_dict(), ternary_best_path)

            torch.save(assignment_list, assignments_path)
            torch.save(opt_sf.param_groups[0]['params'], centroids_path)

        # Logging centroids and their distribution over epochs
        t_weights_value_log, t_weights_counts_log, entropy = give_net_stats(optimizer.param_groups[1]['params'],
                                                                            t_weights_value_log, t_weights_counts_log,
                                                                            entropy, ini_flag=False)
        # Get current sparsity
        sparsity_total, sparsity_quant = calc_sparsity(optimizer, total_params, total_quant_params)
        sparsity_log.append(sparsity_total)

        # Gathering and saving current losses, accuracies
        all_losses += [(epoch, running_loss/n_steps, val_loss, running_accuracy/n_steps, val_accuracy)]
        val_acc.append(val_accuracy * 0.01)

        # Printing epoch's results
        for l in range(optimizer.param_groups[1]['params'].__len__()):
            print('layer: {}, ternary counts: {}\n'.format(l, t_weights_counts_log[l].cpu().detach().numpy()))
            print('layer: {}, w_n value: {}\n'.format(l, t_weights_value_log[l].cpu().detach().numpy()[:, 0]))
            print('layer: {}, w_p value: {}\n'.format(l, t_weights_value_log[l].cpu().detach().numpy()[:, 2]))
        print('lambdas per layer: ', lambda_list)
        print('sparsity total {:.2f}%'.format(sparsity_total * 100))
        print('sparsity of quantized layers {:.2f}%'.format(sparsity_quant * 100))
        print('Epoch {0} running loss {1:.3f} test loss {2:.3f}  running acc {3:.3f} '
              'test acc{4:.3f}  time {5:.3f}'.format(*all_losses[-1], time.time() - start_time))

        # Saving epoch's results in files
        if slurm_save == None:
            logfile = open('./model_quantization/logfiles/logfile.txt', 'a+')
        else:
            logfile = open(slurm_save + '/logfile.txt', 'a+')

        #sparsity_logfile = open('./model_quantization/logfiles/log_per_epoch/sparsity' + str(epoch) + '.txt', 'a+')
        #train_acc_logfile = open('./model_quantization/logfiles/log_per_epoch/train_acc' + str(epoch) + '.txt', 'a+')
        #val_acc_logfile = open('./model_quantization/logfiles/log_per_epoch/val_acc' + str(epoch) + '.txt', 'a+')
        #sparsity_logfile.write('{:.2f}\n'.format(sparsity_total * 100))
        #train_acc_logfile.write('{:.3f}\n'.format(running_accuracy/n_steps))
        #val_acc_logfile.write('{:.3f}\n'.format(val_accuracy))
        logfile.write('sparsity total {:.2f}%\n'.format(sparsity_total * 100))
        logfile.write('sparsity of quantized layers {:.2f}%\n'.format(sparsity_quant * 100))
        logfile.write('Epoch {0} running loss {1:.3f} test loss {2:.3f}  running acc {3:.3f} '
                      'test acc{4:.3f}  time {5:.3f}\n'.format(*all_losses[-1], time.time() - start_time))

        # Resetting variables for next epoch
        running_loss = 0.0
        running_accuracy = 0.0
        n_steps = 0
        start_time = time.time()
        nr_epochs += 1

        # Going back to training mode
        model.train()

    # Plotting
    nr_of_t_layers = optimizer.param_groups[1]['params'].__len__()
    for i in range(nr_of_t_layers):
        plot_weights(i, t_weights_counts_log, t_weights_value_log, entropy, val_acc, sparsity_log,
                     lambda_max_divider, initial_c_scaler, plot_flag, stop_quantizing_flag, nr_of_t_layers)


def calc_intersections(cost, weights):
    """
     ---------------------------------------------------------------
     |Function unused in this code but remains for debugging issues|
     ---------------------------------------------------------------
    Calculates for which layer weights the cost function value is larger for the zero-centroid w_0 than for the
    negative and positive centroid w_n and w_p, i.e. where the algorithm would not assign a given weight to w_0.
    Parameters:
    -----------
        weights:
            Full precision weights of the given layer
        cost:
            Cost list includes the distance of all weights to all centroids added to the information content of all
            centroids, i.e. -log2(probability_centroid_assignment). For the ternary net the list has 3 subspaces.

    Returns:
    --------
        n_intersection:
            Full precision weights that would be assigned to w_n (for a given Lambda)
        p_intersection:
            Full precision weights that would be assigned to w_p (for a given Lambda)
    """
    # Where the cost of w_0 is greater than the cost of w_n, i.e. assignment to w_n
    n_intersection = cost[1].gt(cost[0]).float()
    # Where the cost of w_0 is greater than the cost of w_p, i.e. assignment to w_p
    p_intersection = cost[1].gt(cost[2]).float()
    return n_intersection * weights, p_intersection * weights
