import tifffile 
import os
import glob
import numpy as np
from scipy import ndimage

# Read all files in the current folder
files = glob.glob('*.TIF')
sorted_files = sorted(files)
data = []
# Loop through all files and print their shapes and data types
for file in sorted_files:
    data_file = tifffile.imread(file)
    data.append(data_file)
    print(file, data_file.shape, data_file.dtype)

# Concatenate the left and right pancromatic images
pancromatic_l = np.concatenate((data[0],data[1]), axis=1)
pancromatic_r = np.concatenate((data[2],data[3]), axis=1)
pancromatic = np.concatenate((pancromatic_l,pancromatic_r), axis=0)

# Print the shape and data type of the pancromatic image
print('pancromatic:', pancromatic.shape, pancromatic.dtype)
# Save the pancromatic image
tifffile.imsave('concatenated_pancromatic.tif', pancromatic)

# Remove the last 8 rows and columns of the pancromatic image
cropped_pancromatic = pancromatic[:-8,:-8]
# Print the shape and data type of the pancromatic image
print('pancromatic cropped:', cropped_pancromatic.shape, cropped_pancromatic.dtype)
# Save the pancromatic image
tifffile.imsave('cropped_concatenated_pancromatic.tif', pancromatic)

# Interpolate the pancromatic image 0.25x 
zoomed_pancromatic = ndimage.zoom(cropped_pancromatic, 0.25)

# Print the shape and data type of the zoomed pancromatic image
print('pancromatic zoomed cropped:', zoomed_pancromatic.shape, zoomed_pancromatic.dtype)
# Save the pancromatic image
tifffile.imsave('zoomed_cropped_concatenated_pancromatic.tif', zoomed_pancromatic)