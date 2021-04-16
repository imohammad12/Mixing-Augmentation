import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [1, 0.5, 0])/rgb[...,:3].sum(2)


def get_mask_filter(img):
  attention = rgb2gray(img)
  filter = np.array([attention,attention,attention]).transpose(1,2,0)
  return filter

  
def get_pre_image(orig_img):
  pre_images = orig_img#.permute(0,2,3,1)
#   print(pre_images.shape)
  # plt.imshow(pre_images[0])
  # plt.show()
  return pre_images

def attention_mask_filter(image_batch,payload,showImage=-1):
  mean = payload['mean']
  std = payload['std']
  inv_normalize = transforms.Normalize(
    mean= [-m/s for m, s in zip(mean, std)],
    std= [1/s for s in std]
  )
  inv_tensor = (inv_normalize(image_batch)*255).to(device)
  pre_image = get_pre_image(image_batch)
  outputs, probs, preds = payload['model'].generate_image(pre_image, color=True)
  filters = np.array(list(map(get_mask_filter,outputs)))#np.array([attention,attention,attention]).transpose(1,2,0)
  inv_tensor = (inv_tensor.permute(0,2,3,1)).to(device)
  filtered_image = torch.from_numpy(filters).to(device) * inv_tensor
  if showImage!=-1:
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(np.array(inv_tensor[showImage]).astype('uint8'))
    ax[1].imshow(np.array(filters[showImage]*255).astype('uint8'))
    ax[2].imshow(np.array(filtered_image[showImage]).astype('uint8'))
    plt.show();
  filtered_image = filtered_image.permute(0,3,1,2) / 255
  normalizer = transforms.Normalize(
    mean= mean,
    std= std
  )
  filtered_image = normalizer(filtered_image)
  return filtered_image

def attention_combine_filter(image_batch,payload):
  outputs, probs, preds = payload['model'].generate_image(image_batch, color=True)
  combined_image = [(outputs[i]*0.2+image_batch[i].transpose(1,2,0)*0.8).astype('uint8') for i in range(len(outputs))]
  return combined_image

