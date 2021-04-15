import sys
# put address of the folder I am working with
baseAddr = './filter_code'
sys.path.append(f"{baseAddr }/code")

import torch
import function
import importlib
import matplotlib.pyplot as plt
importlib.reload(function)
from function import attention_combine_filter,attention_mask_filter
from saliency.attribution_methods import CAM
from models import SimpleCNN #, RAN, WideResNetAttention

def test():
  device='cpu'
  # loading model
  weights = torch.load(f"{baseAddr}/model.pth")
  model = SimpleCNN('cifar10', 'CAM').to(device)
  model.load_state_dict(weights['model'])

  #acquire sample image
  # !pip3 install pickle5
  import pickle5 as pickle
  file = open('images_sample.pickle', 'rb')
  sample_image= pickle.load(file) 

  CAM_cifar10 = CAM(model)
  masked_imgs = attention_mask_filter(sample_image,showImage=0,payload={'model':CAM_cifar10,'mean':(0.4914, 0.4822, 0.4465),'std':(0.2023, 0.1994, 0.2010)})
  return masked_imgs
