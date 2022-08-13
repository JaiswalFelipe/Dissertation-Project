!pip install rasterio
!pip install imagecodecs
!pip install tifffile 
!pip install GDAL


import os 
import rasterio
import numpy as np
import tifffile as tiff
import imageio

from skimage import transform
from skimage import img_as_float
from matplotlib import pyplot as plt
from PIL import Image
from rasterio.plot import show
from osgeo import gdal 



def image_patcher(img_name, input_file, output_folder, division): 
  """
  2 OPTIONS: 
  - CSV output for actual coordinates
  - pixel-wise patch naming (default)
  """
  
  data = gdal.Open(input_file)

  gt = data.GetGeoTransform()
  #print(gt) # check actual data information

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

  # Uncomment for OPTION 1
  #xmins = []
  #xmaxs = []
  #ymaxs = []
  #ymins = []
  
  # starting xmax and ymin for pixel-wise patch naming
  cur_x = 250 
  cur_y = 250

  # lists for pixel-wise patching sanity check
  #cur_xs = []
  #cur_xs = []

  for i in range(div):
    for j in range(div):
      xmin = xsteps[i]
      xmax = xsteps[i+1]
      ymax = ysteps[j]
      ymin = ysteps[j+1]

      # pixel-wise coordinates patch-naming
      (x_val, y_val) = (cur_x+i*250, cur_y+j*250)
      
      # sanity check for pixel coordinates
      #cur_xs.append(x_val)
      #cur_ys.append(y_val)

      # Comment if csv output is needed to check actual coordinates
      gdal.Warp(output_folder + str(img_name) + "_" + str(x_val) + "_" + str(y_val) + ".tif",
                data, outputBounds = (xmin, ymin, xmax, ymax), dstNodata = -9999)

      # Print actual coordinates
      #print("xmin: "+str(xmin))
      #print("xmax: "+str(xmax))
      #print("ymin: "+str(ymin))
      #print("ymax: "+str(ymax))
      #print("\n")

      # Uncomment for OPTION 1
      #xmins.append(xmin)
      #xmaxs.append(xmax)
      #ymaxs.append(ymax)
      #ymins.append(ymin)
      
# Uncomment for OPTION 1 
#coords_dict = {'xmins': xmins, 'xmaxs': xmaxs, 'ymaxs': ymaxs, 'ymins': ymins}      
# converting to dataframe
#df = pd.DataFrame(coords_dict)
#df = df.sort_index(ascending=False)
# Creating csv file
#df.to_csv(output_path, index=False)      

# Test run
img_name = 'tile_1599' # use int to identify which map if you have multiple maps for a data folder
img_dir = ''
output_dir = ''
division = 16

image_patcher(img_name, img_dir, output_dir, division)
