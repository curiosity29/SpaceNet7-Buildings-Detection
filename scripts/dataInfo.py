import numpy as np
import pandas as pd
import pyproj as pp
import shapely as sl
import geopandas as gd
from shapely.geometry import shape
import rasterio as rs
import rasterio.warp
import rasterio.mask
import fiona
import matplotlib.pyplot as plt
import rasterio.plot
import cv2
import os
from PIL import Image
import matplotlib.image


pathTif = r"C:\Users\DUNG DO\Downloads\testImg1.tif"
srcImg = np.array(rasterio.open(pathTif).read())

mean = srcImg.mean(axis = 1).mean(axis = 1)


# mean
# array([0.20702111, 0.26308049, 0.24522567])

# std
# array([0.11984661, 0.11106335, 0.09017803])
