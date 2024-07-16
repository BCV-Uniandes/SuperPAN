import numpy as np
from PIL import Image
import tifffile
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

#3424
#3104
parser = argparse.ArgumentParser()
parser.add_argument("--left_patches_folder", type=str, default="patches/left_45_ms_nops_original", help="Path to the folder containing the left patches")
parser.add_argument("--right_patches_folder", type=str, default="patches/right_55_ms_nops_original", help="Path to the folder containing the right patches")
parser.add_argument("--number_of_rows", type=int, default=11, help="Number of rows of patches")
parser.add_argument("--number_of_left_columns", type=int, default=5, help="Number of columns of left patches")
parser.add_argument("--number_of_right_columns", type=int, default=6, help="Number of columns of right patches")
parser.add_argument("--final_image_height", type=int, default=8566*4, help="Height of the final image")
parser.add_argument("--final_image_width", type=int, default=7762*4, help="Width of the final image")
parser.add_argument("--save_name", type=str, default="HR_MS_image.tif", help="Name of the final image")
args = parser.parse_args()

def get_patches_order(img, patch_height, patch_width, threshold):

    # Initialize patch counter
    real_num_patch = 0
    saved_num_patch = 0
    real_order = {}

    # Define img height and width
    img_height = img.shape[1]
    img_width = img.shape[2]

    # Total pixels
    total_pixels = patch_height * patch_width

    for y in tqdm(range(0, img_height, patch_height)):
        for x in range(0, img_width, patch_width):
            # Extract patches
            gt_patch = img[y:y+patch_height, x:x+patch_width]

            # Check class 0 is not over the threshold
            pixels_0 = np.sum(gt_patch == 0)

            if (pixels_0 / total_pixels) < threshold:
                # Save patches as tif files in different folders
                real_order[real_num_patch] = saved_num_patch
                saved_num_patch += 1
            else:
                real_order[real_num_patch] = -1
            real_num_patch += 1

    return real_order

# Load the image from where the patches were extracted
img  = tifffile.imread("datasets/gt/downsampled_rgbned.tif")
real_order = get_patches_order(img, 214, 194, 1)
final_image_array = np.zeros((args.final_image_height, args.final_image_width, 6), dtype=np.uint16)
number_of_columns = args.number_of_left_columns + args.number_of_right_columns
total_patches = args.number_of_rows * number_of_columns
num_patch = 0

start_y = 0
for row in range(args.number_of_rows):
    start_x = 0
    for column in range(number_of_columns):
        if column < args.number_of_left_columns:
            image_path = os.path.join(args.left_patches_folder, f'{num_patch}.tif')
        else:
            image_path = os.path.join(args.right_patches_folder, f'{num_patch}.tif')
        patch = tifffile.imread(image_path)
        final_image_array[start_y:start_y + patch.shape[0], start_x:start_x + patch.shape[1]] = patch
        start_x += patch.shape[1]
        num_patch += 1
    start_y += patch.shape[0]

tifffile.imsave(args.save_name, final_image_array)

# Open the final image with matplotlib and save it in png format
final_image = tifffile.imread(args.save_name).astype(np.uint8)
final_image = final_image[:,:,3:]
plt.imshow(final_image)
plt.axis('off')
plt.savefig(args.save_name.split(".")[0] + ".png")