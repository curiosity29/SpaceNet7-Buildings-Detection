import numpy as np
import geopandas as gd
import rasterio as rs
import rasterio.warp
import rasterio.mask
import matplotlib.pyplot as plt
import rasterio.plot

# path
    # input
pathZip = r"C:\Users\DUNG DO\Downloads\building_label.zip"
pathTif = r"C:\Users\DUNG DO\Downloads\testImg1.tif"
    # output
pathOutImg = r"C:\Test\Building_data\images"
pathOutLabel = r"C:\Test\Building_data\labels"
pathOutSource = r"C:\Test\Building_data\Source"

# read shape and tif file
gdf = gd.read_file(pathZip)
srcImg = rasterio.open(pathTif).read()
# rs.plot.show(src)

# crs of tif file
destination_crs = "EPSG:26910"

# change crs of shape file
new_gdf = gdf.to_crs(epsg = 26910, inplace = False)

shapes = new_gdf["geometry"]

# create mask from shape file
with rs.open(pathTif) as src:
    out_image, out_transform = rs.mask.mask(src, shapes)
    out_meta = src.meta

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

# Cut image multiple rectangular pieces

width = 100
height = 100

# cut original image
tiles_im = [
    srcImg[:, x:x+width,y:y+height] for x in range(0,srcImg.shape[1],width) \
for y in range(0,srcImg.shape[2],height)]

# cut mask image
tiles_mask = [
    out_image[:, x:x+width,y:y+height] for x in range(0,out_image.shape[1],width) \
for y in range(0,out_image.shape[2],height)]
# tiles_im = np.array(tiles_im)
# tiles_mask = np.array(tiles_mask)

## check

# print("input shape: ")
# print((src.shape, out_image.shape))

# print("output shape")
# print((np.array(tiles_im).shape, np.array(tiles_mask).shape))

# rs.plot.show(out_image)
# rs.plot.show(tiles_im[0])
# rs.plot.show(tiles_mask[0])
# print(np.array(tiles_im[0]).shape)
# print(out_image.shape)

## check

path = pathOutSource + "\\masked.tif"

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open(path, "w", **out_meta) as dest:
    dest.write(out_image)

# write to output folder

for idx in range(len(tiles_im)):
    path = pathOutImg + "\\" + str(idx) + ".png"

    with rasterio.open(path, "w", **out_meta) as dest:
        dest.write(tiles_im[idx])


for idx in range(len(tiles_im)):
    path = pathOutLabel + "\\" + str(idx) + ".png"

    with rasterio.open(path, "w", **out_meta) as dest:
        dest.write(tiles_mask[idx])

