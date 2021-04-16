#!/usr/bin/env python
# coding: utf-8

# In[25]:


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
# import cv2 as cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from . import surrounding_augmentation as sn
from . import environment_helpers as hp
from random import randrange
import pickle
from PIL import Image
#!pip install cifar2png
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def augment_environment(image_batch,payload,showImage=-1):
 mean = payload['mean']
 std = payload['std']
 inv_normalize = transforms.Normalize(
    mean= [-m/s for m, s in zip(mean,std)],
    std= [1/s for s in std]
  )
 inv_tensor = inv_normalize(image_batch).to(device) * 255
 inv_tensor = inv_tensor.permute(0,2,3,1).to(device)
 img_arr=inv_tensor.cpu().numpy().astype('uint8')
 aug_list=[]
 for i in range(len(img_arr)):
  aug_images= sn.augment_random(img_arr[i], aug_types=['random_brightness','add_gravel','add_snow','add_rain','add_autumn','random_flip','random_brightness'], volume='expand')  ##all aug_types are applied in both images
  rand_image=aug_images[randrange(7)]
  if showImage != -1:
    plt.imshow(rand_image)
    plt.show()
  rand_image=torch.from_numpy(np.transpose(rand_image, (2, 0, 1))).float()/255
  normalizer = transforms.Normalize(
    mean= mean,
    std= std
   )
  new_image=normalizer(rand_image)
  aug_list.append(new_image)
 aug_batch=torch.stack(aug_list)
 return aug_batch






