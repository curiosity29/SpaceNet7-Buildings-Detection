import numpy as np
import rasterio as rs

pathTif = r"C:\Users\DUNG DO\Downloads\testImg1.tif"
srcImg = np.array(rs.open(pathTif).read())

mean = srcImg.mean(axis = (1,2))/255
std = srcImg.std(axis = (1,2))/255
print(mean, std)

# mean
# array([0.20702111, 0.26308049, 0.24522567])

# std
# array([0.11984661, 0.11106335, 0.09017803])

# image-net mean and std: 
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
