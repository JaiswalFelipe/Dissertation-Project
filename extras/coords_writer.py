# -*- coding: utf-8 -*-
"""coords_writer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QqsKPpuXmqEsQH0hfksHYGn4OwKoDdE4
"""

!pip install rasterio
!pip install imagecodecs
!pip install tifffile 
!pip install GDAL


import os 
import rasterio
import numpy as np
import tifffile as tiff
import imageio
import pandas as pd

from skimage import transform
from skimage import img_as_float
from matplotlib import pyplot as plt
from PIL import Image
from rasterio.plot import show
from osgeo import gdal 


# Settings
image_path = ''
division = 16 # divisor of the whole image 
cur_map = np.zeros(256).astype(int) # to identify current map (for validation pipeline purposes)
output_path = '' # include csv name


# patcher
data = gdal.Open(image_path)

gt = data.GetGeoTransform()
    #print(gt)

xmin = gt[0]
ymax = gt[3]
res = gt[1]

xlen = res * data.RasterXSize
ylen = res * data.RasterYSize

div = division

xsize = xlen/div
ysize = ylen/div

xsteps = [xmin + xsize * i for i in range(div+1)]
ysteps = [ymax - ysize * i for i in range(div+1)]

# storing values into the lists
#xmins = []
xmaxs = []
#ymaxs = []
ymins = []

for i in range(div):
  for j in range(div):
    xmin = xsteps[i]
    xmax = xsteps[i+1]
    ymax = ysteps[j]
    ymin = ysteps[j+1]

    #xmins.append(xmin)
    xmaxs.append(xmax)
    #ymaxs.append(ymax)
    ymins.append(ymin)


# Getting a dictionary for the required corrdinates 
coords_dict = {'cur_map': cur_map, 'cur_x': xmaxs, 'cur_y': ymins}

# converting to dataframe
df = pd.DataFrame(coords_dict)

# Creating csv file
df.to_csv(output_path, index=False)