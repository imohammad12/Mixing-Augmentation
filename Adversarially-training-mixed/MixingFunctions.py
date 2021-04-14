#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# import models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
#from tqdm import tqdm

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# parser.add_argument('--model', default="ResNet18", type=str,
#                     help='model type (default: ResNet18)')
# parser.add_argument('--name', default='0', type=str, help='name of run')
# parser.add_argument('--seed', default=0, type=int, help='random seed')
# parser.add_argument('--batch-size', default=128, type=int, help='batch size')
# parser.add_argument('--epoch', default=200, type=int,
#                     help='total epochs to run')
# parser.add_argument('--no-augment', dest='augment', action='store_false',
#                     help='use standard augmentation (default: True)')
# parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
# parser.add_argument('--alpha', default=1., type=float,
#                     help='mixup interpolation coefficient (default: 1)')
# args = parser.parse_args()

params = {
    "lr": .01,
    "resume": False,
    "model": "ResNet18",
    "name": "mixup-128-NormalAdvsTrain",
    "seed": 10,
    "batch_size": 128,
    "decay": 5e-4, 
    "augment": True,
    "epoch": 200,
#     "no_augment": False,
    "alpha": 0.,
}



use_cuda = torch.cuda.is_available()


# # Helping functions

# In[18]:


def shuffle_dat(x, batch_size, use_cuda):
    '''shuffles data of batch size'''
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
        
    return x[index,:], index

def mixup_data_nMixes(x, y, nMix, net, criterion, use_cuda=True, epsilon=.35, advs_train=False):
    '''Returns mixed images for general number of mixings, targets of mixings, and mixing weights'''
    
    # Create weights in range [0,1], and sum to 1.0
    wts = np.random.dirichlet(np.ones(nMix))
    
    # Create tensor to hold all shuffled targets, has size (batch_size, nMix)
    if use_cuda:
        mixed_targs = torch.zeros(y.size()[0], nMix).cuda()
        mixed_indices = torch.zeros(nMix, x.size()[0]).cuda()
    else:
        mixed_targs = torch.zeros(y.size()[0], nMix)
        mixed_indices = torch.zeros(nMix, x.size()[0])
    
    batch_size = x.size()[0]
    
    # Initiate mixed x as original x with first weight, and first col of targets to be original targets
    mixed_x = 0*x
    mixed_noise = 0 * x
    
    if advs_train:
        x_noise = fgsm(x, y, net, criterion, epsilon)
        mixed_noise += wts[0]*x_noise
    
    mixed_x += wts[0]*x
    mixed_targs[:,0] = y
    mixed_indices[0,:] = torch.arange(0,batch_size)
    
    # Add shuffled images to mixed image with weights, accumulate mixed targets
    for i in range(1, nMix):
        shuffledX, index = shuffle_dat(x, batch_size, use_cuda) 
        mixed_x += wts[i]*shuffledX
        
        if advs_train:
            full_advs_noise = fgsm(shuffledX, y[index], net, criterion, epsilon)
            mixed_noise += wts[i] * full_advs_noise
        
        mixed_targs[:,i] = y[index]
        mixed_indices[i,:] = index
    
    return mixed_x, mixed_targs, mixed_noise, wts

def vertical_concat_mix(x, y, nMix, net, criterion, use_cuda=True, epsilon=.35, advs_train=False):
    '''Returns mixed image for general number of vertical concats'''
    # Create weights in range [0,1], and sum to 1.0
    wts = np.random.dirichlet(np.ones(nMix))
    
    # Create tensor to hold all shuffled targets, has size (batch_size, nMix)
    if use_cuda:
        mixed_targs = torch.zeros(y.size()[0], nMix).cuda()
    else:
        mixed_targs = torch.zeros(y.size()[0], nMix)
        
    batch_size = x.size()[0]
    num_rows = x.size()[2]
    
    # Initiate mixed image as first n rows of original image, accoring to the first weight, first set of targets are
    # original targets
    

    mixed_noise = 0 * x
    
    if advs_train:
        x_noise = fgsm(x, y, net, criterion, epsilon)
        mixed_noise[:,:,0:int(round(wts[0]*num_rows)),:] = x_noise[:,:,0:int(round(wts[0]*num_rows)),:]
    
    mixed_x = 0 * x
    mixed_x[:,:,0:int(round(wts[0]*num_rows)),:] = x[:,:,0:int(round(wts[0]*num_rows)),:]
    
    mixed_targs[:,0] = y
    start_row = int(round(wts[0]*num_rows))
    
    # Add subsequent number of rows to mixed image, accoring to weights, accumulate shuffled targets
    for i in range(1, nMix-1):
        shuffledX, index = shuffle_dat(x, batch_size, use_cuda)
        mixed_x[:,:,start_row:start_row + int(round(wts[i]*num_rows)),:] =  \
                                      shuffledX[:,:,start_row:start_row + int(round(wts[i]*num_rows)),:] 
                
        if advs_train:
            full_advs_noise = fgsm(shuffledX, y[index], net, criterion, epsilon)
            mixed_noise[:,:,start_row:start_row + int(round(wts[i]*num_rows)),:] = full_advs_noise[:,:,start_row:start_row + int(round(wts[i]*num_rows)),:] 
        
        start_row += int(round(wts[i]*num_rows))
        mixed_targs[:,i] = y[index]
    
    # Finish of mixed image with final concatenation, add final mixed targets
    shuffledX, index = shuffle_dat(x, batch_size, use_cuda)
    mixed_x[:,:,start_row:,:] = shuffledX[:,:,start_row:,:] 
    
    if advs_train:
        full_advs_noise = fgsm(shuffledX, y[index], net, criterion, epsilon)
        mixed_noise[:,:,start_row:,:] =  full_advs_noise[:,:,start_row:,:] 
    
    mixed_targs[:,-1] = y[index]
        
    return mixed_x, mixed_targs, mixed_noise, wts   



def horizontal_concat_mix(x, y, nMix, net, criterion, use_cuda=True, epsilon=.35, advs_train=False):
    '''Returns mixed image for general number of horizontal concats''' 
    # Works the same as vertical_concat mix, but switches dimenions of concats.
    wts = np.random.dirichlet(np.ones(nMix))
    
    if use_cuda:
        mixed_targs = torch.zeros(y.size()[0], nMix).cuda()
        mixed_indices = torch.zeros(nMix, x.size()[0]).cuda()
    else:
        mixed_targs = torch.zeros(y.size()[0], nMix)
        mixed_indices = torch.zeros(nMix, x.size()[0])
        
    batch_size = x.size()[0]
    num_cols = x.size()[3]
    
    
    mixed_noise = 0 * x
    
    if advs_train:
        x_noise = fgsm(x, y, net, criterion, epsilon)
        mixed_noise[:,:,:,0:int(round(wts[0]*num_cols))] = x_noise[:,:,:,0:int(round(wts[0]*num_cols))]
    
    mixed_x = 0*x
    mixed_x[:,:,:,0:int(round(wts[0]*num_cols))] = x[:,:,:,0:int(round(wts[0]*num_cols))]
    
    mixed_targs[:,0] = y
    mixed_indices[0,:] = torch.arange(0,batch_size)
    start_col = int(round(wts[0]*num_cols))
    
    for i in range(1, nMix-1):
        shuffledX, index = shuffle_dat(x, batch_size, use_cuda)
        mixed_x[:,:,:,start_col:start_col + int(round(wts[i]*num_cols))] =  \
                                      shuffledX[:,:,:,start_col:start_col + int(round(wts[i]*num_cols))] 
        
        if advs_train:
            full_advs_noise = fgsm(shuffledX, y[index], net, criterion, epsilon)
            mixed_noise[:,:,:,start_col:start_col + int(round(wts[i]*num_cols))] = full_advs_noise[:,:,:,start_col:start_col + int(round(wts[i]*num_cols))]
        
        start_col += int(round(wts[i]*num_cols))
        mixed_targs[:,i] = y[index]
        mixed_indices[i,:] = index
        
    shuffledX, index = shuffle_dat(x, batch_size, use_cuda)
    mixed_x[:,:,:,start_col:] =  \
                                      shuffledX[:,:,:,start_col:] 
    
    if advs_train:
        full_advs_noise = fgsm(shuffledX, y[index], net, criterion, epsilon)
        mixed_noise[:,:,:,start_col:] = full_advs_noise[:,:,:,start_col:]
    
    
    mixed_targs[:,-1] = y[index]
    mixed_indices[-1,:] = index
        
    return mixed_x, mixed_targs, mixed_noise, wts   


def mixup_criterion_nMixes(criterion, pred, mixed_targs, wts):
    '''Returns total criterion for total mixed images'''
    
    # accumulate total criterion as weighted sum of shuffled targets, weighted with wts 
    total_criterion = 0
    for i in range(len(wts)):
        total_criterion += wts[i]*criterion(pred, mixed_targs[:,i])
    return(total_criterion)

def fgsm(x, y, net, criterion,  eps=.35):
    
    x = x.detach()
    x.requires_grad_(True)
    out = net(x)
    loss = criterion(out, y)
    net.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        err = eps * x.grad.sign()    
    return err

# def mixup_data_nMixes(x, y, nMix, use_cuda=True, epsilon=.35):
#     '''Returns mixed images for general number of mixings, targets of mixings, and mixing weights'''
    
#     # Create weights in range [0,1], and sum to 1.0
#     wts = np.random.dirichlet(np.ones(nMix))
    
#     # Create tensor to hold all shuffled targets, has size (batch_size, nMix)
#     if use_cuda:
#         mixed_targs = torch.zeros(y.size()[0], nMix).cuda()
#         mixed_indices = torch.zeros(nMix, x.size()[0]).cuda()
#     else:
#         mixed_targs = torch.zeros(y.size()[0], nMix)
#         mixed_indices = torch.zeros(nMix, x.size()[0])
    
#     batch_size = x.size()[0]
    
#     # Initiate mixed x as original x with first weight, and first col of targets to be original targets
#     mixed_x = 0*x
#     mixed_x += wts[0]*x
#     mixed_targs[:,0] = y
#     mixed_indices[0,:] = torch.arange(0,batch_size)
    
#     # Add shuffled images to mixed image with weights, accumulate mixed targets
#     for i in range(1, nMix):
#         shuffledX, index = shuffle_dat(x, batch_size, use_cuda) 
#         mixed_x += wts[i]*shuffledX
#         mixed_targs[:,i] = y[index]
#         mixed_indices[i,:] = index
    
#     return mixed_x, mixed_targs, mixed_indices, wts

# def vertical_concat_mix(x, y, nMix, use_cuda=True, epsilon=.35):
#     '''Returns mixed image for general number of vertical concats'''
#     # Create weights in range [0,1], and sum to 1.0
#     wts = np.random.dirichlet(np.ones(nMix))
    
#     # Create tensor to hold all shuffled targets, has size (batch_size, nMix)
#     if use_cuda:
#         mixed_targs = torch.zeros(y.size()[0], nMix).cuda()
#         mixed_indices = torch.zeros(nMix, x.size()[0]).cuda()
#     else:
#         mixed_targs = torch.zeros(y.size()[0], nMix)
#         mixed_indices = torch.zeros(nMix, x.size()[0])
        
#     batch_size = x.size()[0]
#     num_rows = x.size()[2]
    
#     # Initiate mixed image as first n rows of original image, accoring to the first weight, first set of targets are
#     # original targets
#     mixed_x = 0*x
#     mixed_x[:,:,0:int(round(wts[0]*num_rows)),:] = x[:,:,0:int(round(wts[0]*num_rows)),:]
#     mixed_targs[:,0] = y
#     mixed_indices[0,:] = torch.arange(0,batch_size)
#     start_row = int(round(wts[0]*num_rows))
    
#     # Add subsequent number of rows to mixed image, accoring to weights, accumulate shuffled targets
#     for i in range(1, nMix-1):
#         shuffledX, index = shuffle_dat(x, batch_size, use_cuda)
#         mixed_x[:,:,start_row:start_row + int(round(wts[i]*num_rows)),:] =                                        shuffledX[:,:,start_row:start_row + int(round(wts[i]*num_rows)),:] 
#         start_row += int(round(wts[i]*num_rows))
#         mixed_targs[:,i] = y[index]
#         mixed_indices[i,:] = index
    
#     # Finish of mixed image with final concatenation, add final mixed targets
#     shuffledX, index = shuffle_dat(x, batch_size, use_cuda)
#     mixed_x[:,:,start_row:,:] =                                        shuffledX[:,:,start_row:,:] 
#     mixed_targs[:,-1] = y[index]
#     mixed_indices[-1,:] = index
        
#     return mixed_x, mixed_targs, mixed_indices, wts   

# def horizontal_concat_mix(x, y, nMix, use_cuda=True, epsilon=.35):
#     '''Returns mixed image for general number of horizontal concats''' 
#     # Works the same as vertical_concat mix, but switches dimenions of concats.
#     wts = np.random.dirichlet(np.ones(nMix))
    
#     if use_cuda:
#         mixed_targs = torch.zeros(y.size()[0], nMix).cuda()
#         mixed_indices = torch.zeros(nMix, x.size()[0]).cuda()
#     else:
#         mixed_targs = torch.zeros(y.size()[0], nMix)
#         mixed_indices = torch.zeros(nMix, x.size()[0])
        
#     batch_size = x.size()[0]
#     num_cols = x.size()[3]
    
#     mixed_x = 0*x
#     mixed_x[:,:,:,0:int(round(wts[0]*num_cols))] = x[:,:,:,0:int(round(wts[0]*num_cols))]
#     mixed_targs[:,0] = y
#     mixed_indices[0,:] = torch.arange(0,batch_size)
#     start_col = int(round(wts[0]*num_cols))
    
#     for i in range(1, nMix-1):
#         shuffledX, index = shuffle_dat(x, batch_size, use_cuda)
#         mixed_x[:,:,:,start_col:start_col + int(round(wts[i]*num_cols))] =                                        shuffledX[:,:,:,start_col:start_col + int(round(wts[i]*num_cols))] 
#         start_col += int(round(wts[i]*num_cols))
#         mixed_targs[:,i] = y[index]
#         mixed_indices[i,:] = index
        
#     shuffledX, index = shuffle_dat(x, batch_size, use_cuda)
#     mixed_x[:,:,:,start_col:] =                                        shuffledX[:,:,:,start_col:] 
#     mixed_targs[:,-1] = y[index]
#     mixed_indices[-1,:] = index
        
#     return mixed_x, mixed_targs, mixed_indices, wts   


# def mixup_criterion_nMixes(criterion, pred, mixed_targs, wts):
#     '''Returns total criterion for total mixed images'''
    
#     # accumulate total criterion as weighted sum of shuffled targets, weighted with wts 
#     total_criterion = 0
#     for i in range(len(wts)):
#         total_criterion += wts[i]*criterion(pred, mixed_targs[:,i])
#     return(total_criterion)



