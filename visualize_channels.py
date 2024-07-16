import tifffile as tiff
import numpy as np
from skimage.transform import rescale
import matplotlib.pyplot as plt

# Read images
RGB_image_path = "CombinadoRGB.tif"
NED_image_path = "CombinadoNED.tif"
RGBNED_image_path = "mixed_rgb_ned/merged.tif"

#RGB_img = tiff.imread(RGB_image_path).astype(np.uint8)
#NED_img = tiff.imread(NED_image_path).astype(np.uint8)
RGBNED_img = tiff.imread(RGBNED_image_path).astype(np.uint8)

breakpoint()

"""
# Visualizar cada canal en RGB image en color
for i in range(3):
    img = np.zeros_like(RGB_img)
    img[:,:,i] = RGB_img[:,:,i]
    tiff.imwrite(f"RGB_channel_color_{i}.png", img, compression="zlib")"""

# Visualizar cada canal en NED image en color
for i in range(3):
    img = NED_img[:,:,i]
    if i == 0:
        cmap = "cividis"
    elif i == 1:
        cmap = "inferno"
    else:
        cmap = "Blues_r"
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=cmap) 
    plt.colorbar()  
    plt.show()
    plt.savefig(f'NED_channel_{i}.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
    #tiff.imwrite(f"NED_channel_color_{i}.png", img, compression="zlib")