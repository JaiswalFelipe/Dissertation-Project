import re
import os
import sys
import numpy as np
import pandas as pd

import imageio
from skimage import transform
from skimage import img_as_float

import torch
from torch.utils import data

from data_utils import create_or_load_statistics, create_distrib, normalize_images, data_augmentation, compute_image_mean


# training_dataset class
class NGTrain(data.Dataset):
  def __init__(self, img_dir, mask_dir, output_path):

    #self.dataset_input_path = dataset_input_path
    #self.images = images
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.images = os.listdir(img_dir)
    self.masks = os.listdir(mask_dir)

    self.output_path = output_path


    # data and label
    self.data, self.labels = self.load_images()

    print(self.data.ndim, self.data.shape, self.data[0].shape, np.min(self.data), np.max(self.data),
          self.labels.shape, np.bincount(self.labels.astype(int).flatten()))

    if self.data.ndim == 4:  # if all images have the same shape
       self.num_channels = self.data.shape[-1]  # get the number of channels
    else:
       self.num_channels = self.data[0].shape[-1]  # get the number of channels

    self.num_classes = 2  # binary - two classes
    # negative classes will be converted into 2 so they can be ignored in the loss
    self.labels[np.where(self.labels < 0)] = 2
    
    print('num_channels and labels', self.num_channels, self.num_classes, np.bincount(self.labels.flatten()))

    #self.distrib, self.gen_classes = self.make_dataset()

    self.mean, self.std = compute_image_mean(self.data)


  def load_images(self):
        images = []
        masks = []
        for img in self.images:
            temp_image = imageio.imread(os.path.join(self.img_dir, img + '')).astype(np.float64)
            temp_image[np.where(temp_image < -1.0e+38)] = 0 # remove extreme negative values (probably NO_DATA values)
            
            images.append(temp_image)

        for msk in self.masks:
            temp_mask = imageio.imread(os.path.join(self.mask_dir, msk + '')).astype(int)
            temp_mask[np.where(temp_mask < -1.0e+38)] = 0

            masks.append(temp_mask)

        return np.asarray(images), np.asarray(masks)


  def __getitem__(self, index):
    
    #Reading items from list.
    img = self.data[index]
    label = self.labels[index]

    # Normalization.
    normalize_images(img, self.mean, self.std) # check data_utils.py
    
    # Data augmentation
    img, label = data_augmentation(img, label)
     
    img = np.transpose(img, (2, 0, 1))

    # Turning to tensors.
    img = torch.from_numpy(img.copy())
    label = torch.from_numpy(label.copy())

    # Returning to iterator.
    return img.float(), label

  def __len__(self):
    return len(self.data)



# testing_dataset class
class NGTest(data.Dataset):
    def __init__(self, dataset_input_path, images, crop_size, stride_crop, output_path):
        #super().__init__()
        #assert mode in ['Train', 'Test']

        #self.mode = mode
        self.dataset_input_path = dataset_input_path
        self.images = images
        self.crop_size = crop_size
        self.stride_crop = stride_crop

        self.output_path = output_path

        # data and label
        self.data, self.labels = self.load_images()
        #self.data[np.where(self.data < -1.0e+38)] = 0  # remove extreme negative values (probably NO_DATA values)
        print(self.data.ndim, self.data.shape, self.data[0].shape, np.min(self.data), np.max(self.data),
              self.labels.shape, np.bincount(self.labels.astype(int).flatten()))

        if self.data.ndim == 4:  # if all images have the same shape
            self.num_channels = self.data.shape[-1]  # get the number of channels
        else:
            self.num_channels = self.data[0].shape[-1]  # get the number of channels

        self.num_classes = 2  # binary - two classes
        # negative classes will be converted into 2 so they can be ignored in the loss
        self.labels[np.where(self.labels < 0)] = 2

        print('num_channels and labels', self.num_channels, self.num_classes, np.bincount(self.labels.flatten()))

        self.distrib, self.gen_classes = self.make_dataset()

        self.mean, self.std = create_or_load_statistics(self.data, self.distrib, self.crop_size,
                                                        self.stride_crop, self.output_path)

        if len(self.distrib) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    def load_images(self):
        images = []
        masks = []
        for img in self.images:
            temp_image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path, img + '_stack.tif')))
            temp_image[np.where(temp_image < -1.0e+38)] = 0  # remove extreme negative values (probably NO_DATA values)  

            temp_mask = imageio.imread(os.path.join(self.dataset_input_path, img + '_mask.tif')).astype(int)
            temp_mask[np.where(temp_mask < -1.0e+38)] = 0
            
            images.append(temp_image)
            masks.append(temp_mask)

        return np.asarray(images), np.asarray(masks)

    def make_dataset(self):
        return create_distrib(self.labels, self.crop_size, self.stride_crop, self.num_classes, return_all=True)

    def __getitem__(self, index):
        # Reading items from list.
        cur_map, cur_x, cur_y = self.distrib[index][0], self.distrib[index][1], self.distrib[index][2]

        img = np.copy(self.data[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size, :])
        label = np.copy(self.labels[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size])

        # Normalization.
        normalize_images(img, self.mean, self.std)

        #if self.mode == 'Train':
        #    img, label = data_augmentation(img, label)

        img = np.transpose(img, (2, 0, 1))

        # Turning to tensors.
        img = torch.from_numpy(img.copy())
        label = torch.from_numpy(label.copy())

        # Returning to iterator.
        return img.float(), label, cur_map, cur_x, cur_y

    def __len__(self):
        return len(self.distrib)




# validation_dataset class
# needs a separate class because in validation, we load patches instead of whole images
# along with their coordinates for reconstruction
class NGValid(data.Dataset):
  def __init__(self, img_dir, mask_dir, output_path):

    #self.dataset_input_path = dataset_input_path
    #self.images = images
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.images = os.listdir(img_dir)
    self.masks = os.listdir(mask_dir)

    self.output_path = output_path


    # data and label
    self.data, self.labels, self.cur_maps, self.cur_xs, self.cur_ys = self.load_images()

    print(self.data.ndim, self.data.shape, self.data[0].shape, np.min(self.data), np.max(self.data),
          self.labels.shape, np.bincount(self.labels.astype(int).flatten()))

    if self.data.ndim == 4:  # if all images have the same shape
       self.num_channels = self.data.shape[-1]  # get the number of channels
    else:
       self.num_channels = self.data[0].shape[-1]  # get the number of channels

    self.num_classes = 2  # binary - two classes
    # negative classes will be converted into 2 so they can be ignored in the loss
    self.labels[np.where(self.labels < 0)] = 2
    
    print('num_channels and labels', self.num_channels, self.num_classes, np.bincount(self.labels.flatten()))

    #self.distrib, self.gen_classes = self.make_dataset()

    self.mean, self.std = compute_image_mean(self.data)


  def load_images(self):
        images = []
        cur_maps = []
        cur_xs = []
        cur_ys = []
        masks = []
        for img in self.images:
            temp_image = imageio.imread(os.path.join(self.img_dir, img + '')).astype(np.float64)
            temp_image[np.where(temp_image < -1.0e+38)] = 0 # remove extreme negative values (probably NO_DATA values)
            
            # Extracting coordinates
            fn_parse = str(img.replace('.tif', ''))
            cur_map = str(re.split("_", fn_parse)[0])
            cur_x = str(re.split("_", fn_parse)[1])
            cur_y = str(re.split("_", fn_parse)[-1])

            cur_maps.append(cur_map)
            cur_xs.append(cur_x)
            cur_ys.append(cur_y)
            images.append(temp_image)

        for msk in self.masks:
            temp_mask = imageio.imread(os.path.join(self.mask_dir, msk + '')).astype(int)
            temp_mask[np.where(temp_mask < -1.0e+38)] = 0

            masks.append(temp_mask)

        return np.asarray(images), np.asarray(masks), cur_maps, cur_xs, cur_ys


  def __getitem__(self, index):
    
    #Reading items from list.
    cur_map = self.cur_maps[index]
    cur_x = self.cur_xs[index]
    cur_y = self.cur_ys[index]
    
    img = np.copy(self.data[cur_map][cur_x, cur_y, :])
    label = np.copy(self.labels[cur_map][cur_x, cur_y])

    # Normalization.
    normalize_images(img, self.mean, self.std) # check data_utils.py
    
    # Data augmentation
    #img, label = data_augmentation(img, label)
     
    img = np.transpose(img, (2, 0, 1))

    # Turning to tensors.
    img = torch.from_numpy(img.copy())
    label = torch.from_numpy(label.copy())

    # Returning to iterator.
    return img.float(), label, cur_map, cur_x, cur_y
  
  def __len__(self):
    return len(self.data)
