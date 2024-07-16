import matplotlib.pyplot as plt
import tiffile as tiff
import numpy as np
import shutil
import cv2
import os
import argparse

# FIXME: Add arguments

parser = argparse.ArgumentParser()
parser.add_argument("--shp_path", type=str, help="Path to the shp file")
parser.add_argument("--rgb_path", type=str, help="Path to the RGB image")
parser.add_argument("--ned_path", type=str, help="Path to the NED image")
parser.add_argument("--pancromatic_path", type=str, help="Path to the pancromatic image")
parser.add_argument("--output_path", type=str, help="Path to the output image")
args = parser.parse_args()

if not os.path.exists(args.output_path):
    # Concatenate shp, RGB and NED images with gdal to conserve geotransform parameters and spatial reference system
    os.system(f"gdal_merge.py -separate -o {args.output_path} {args.shp_path} {args.rgb_path} {args.ned_path}")

# Read shp image as tiff
shp = tiff.imread(args.shp_path)
# Expand dimensions to add a new channel
shp = np.expand_dims(shp, axis=2)
# Read RGB image
rgb = tiff.imread(args.rgb_path)
# Read NED image
ned = tiff.imread(args.ned_path)
# Read combinado RGB_NED_shp image
rgb_ned_shp = tiff.imread(args.output_path)

# Interpolated image
LR_image_path = "datasets/gt/downsampled_rgbned.tif"
# Orignal image
HR_image_path = "datasets/gt/modified_rgbned.tif"
# Pancromatic image
pancromatic_image_path = "datasets/gt/downsampled_PAN.tif"

# Read images and omit last row of LR image, last 4 rows of HR image and last 4 rows of pancromatic image
LR_img = tiff.imread(LR_image_path).astype(np.uint16)[:,:-1,:]
HR_img = tiff.imread(HR_image_path).astype(np.uint16)[:,:-4,:]
Pancromatic_img = tiff.imread(pancromatic_image_path).astype(np.uint16)[:-4,:]

# Split images in half
fraction = 0.45
LR_img_left = LR_img[:,:,:int(LR_img.shape[2]*fraction)] 
LR_img_right = LR_img[:,:,int(LR_img.shape[2]*fraction):] 
HR_img_left = HR_img[:,:,:int(HR_img.shape[2]*fraction)] 
HR_img_right = HR_img[:,:,int(HR_img.shape[2]*fraction):] 
P_img_left = Pancromatic_img[:,:int(Pancromatic_img.shape[1]*fraction)] 
P_img_right = Pancromatic_img[:,int(Pancromatic_img.shape[1]*fraction):] 

def plot_image(img, save_path):
    # Transpose the image
    img = np.transpose(img, (1,2,0))
    # Convert the data range to [0, 255] for visualization
    img = (img / np.iinfo('uint16').max * 255).astype('uint8')
    # Get the first 3 channels
    img = img[:,:,:3]
    # Convert RGB to HSV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Equalize the value channel
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    # Convert back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    # Calculate histogram
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    # Plot image
    plt.figure()
    plt.plot(histogram)
    plt.grid(False)
    #plt.xticks([])
    #plt.yticks([])
    plt.title(save_path)
    plt.savefig(save_path+".png")

# Visualize images
plot_image(LR_img_left, "hist_LR_image_left_0.45")
plot_image(LR_img_right, "hist_LR_image_right_0.55")
plot_image(HR_img_left, "hist_HR_image_left_0.45")
plot_image(HR_img_right, "hist_HR_image_right_0.55")
breakpoint()

# Create patches no overlaped and save them
def get_patches_no_overlaped(LR_img, HR_img, P_img, patch_height, patch_width, exp_name):

    # Create folders if they don't exist
    if not os.path.exists(f"patches/all_LR_{exp_name}"):
        os.makedirs(f"patches/all_LR_{exp_name}")
    if not os.path.exists(f"patches/all_HR_{exp_name}"):
        os.makedirs(f"patches/all_HR_{exp_name}")
    if not os.path.exists(f"patches/all_Pancromatic_{exp_name}"):
        os.makedirs(f"patches/all_Pancromatic_{exp_name}")

    # Create set to store different patch sizes
    unique_lr_shapes = set()
    unique_hr_shapes = set()
    unique_p_shapes = set()

    # Initialize patch counter
    num_patch = 0

    for y in range(0, LR_img.shape[1], patch_height): # dividir entre el height
        for x in range(0, LR_img.shape[2], patch_width): # dividir entre 2 el width
            # Extract patches
            LR_patch = LR_img[:, y:y+patch_height, x:x+patch_width]
            HR_patch = HR_img[:, y*4:(y+patch_height)*4, x*4:(x+patch_width)*4]
            Pancromatic_patch = P_img[y*4:(y+patch_height)*4, x*4:(x+patch_width)*4] # 1 channel in high resolution

            # Check if patch values are not only 0 for LR and HR patches 
            if LR_patch.max() != 0 and HR_patch.max() != 0:
                # Verificación de que no sea de la mitad del tamaño
                if LR_patch.shape[1]==patch_height and LR_patch.shape[2]==patch_width:
                    # Store different patch sizes
                    unique_lr_shapes.add(LR_patch.shape)
                    unique_hr_shapes.add(HR_patch.shape)
                    unique_p_shapes.add(Pancromatic_patch.shape)

                    # Save patches as tif files in different folders and convert to uint16
                    tiff.imwrite(f"patches/all_LR_{exp_name}/{num_patch}.tif", LR_patch.astype(np.uint16)) # agregarle left or right
                    tiff.imwrite(f"patches/all_HR_{exp_name}/{num_patch}.tif", HR_patch.astype(np.uint16))
                    tiff.imwrite(f"patches/all_Pancromatic_{exp_name}/{num_patch}.tif", Pancromatic_patch.astype(np.uint16))
                    num_patch += 1
    
    # Print different patch sizes
    print("Patches:", num_patch)
    print(f"Different LR patch sizes:", unique_lr_shapes)
    print(f"Different HR patch sizes:", unique_hr_shapes)
    print(f"Different Pancromatic patch sizes:", unique_p_shapes)


# Create patches overlaped and save them
def get_patches_overlaped(LR_img, HR_img, P_img, patch_height, patch_width, exp_name):

    # Create folders if they don't exist
    if not os.path.exists(f"patches/all_LR_{exp_name}"):
        os.makedirs(f"patches/all_LR_{exp_name}")
    if not os.path.exists(f"patches/all_HR_{exp_name}"):
        os.makedirs(f"patches/all_HR_{exp_name}")
    if not os.path.exists(f"patches/all_Pancromatic_{exp_name}"):
        os.makedirs(f"patches/all_Pancromatic_{exp_name}")

    # Create set to store different patch sizes
    unique_lr_shapes = set()
    unique_hr_shapes = set()
    unique_p_shapes = set()

    # Initialize patch counter
    num_patch = 0

    for y in range(0, LR_img.shape[1], patch_height//2): 
        for x in range(0, LR_img.shape[2], patch_width//2): 

            # Extract patches
            LR_patch = LR_img[:, y:y+patch_height, x:x+patch_width]
            HR_patch = HR_img[:, y*4:(y+patch_height)*4, x*4:(x+patch_width)*4]
            Pancromatic_patch = P_img[y*4:(y+patch_height)*4, x*4:(x+patch_width)*4] # 1 channel in high resolution

            # Check if patch values are not only 0 for LR and HR patches 
            if LR_patch.max() != 0 and HR_patch.max() != 0:
                # Verificación de que no sea de la mitad del tamaño
                if LR_patch.shape[1]==patch_height and LR_patch.shape[2]==patch_width:
                    # Store different patch sizes
                    unique_lr_shapes.add(LR_patch.shape)
                    unique_hr_shapes.add(HR_patch.shape)
                    unique_p_shapes.add(Pancromatic_patch.shape)

                    # Save patches as tif files in different folders and convert to uint16
                    tiff.imwrite(f"patches/all_LR_{exp_name}/{num_patch}.tif", LR_patch.astype(np.uint16)) # agregarle left or right
                    tiff.imwrite(f"patches/all_HR_{exp_name}/{num_patch}.tif", HR_patch.astype(np.uint16))
                    tiff.imwrite(f"patches/all_Pancromatic_{exp_name}/{num_patch}.tif", Pancromatic_patch.astype(np.uint16))
                    num_patch += 1
    
    # Print different patch sizes
    print("Patches:", num_patch)
    print("Different LR patch sizes:", unique_lr_shapes)
    print("Different HR patch sizes:", unique_hr_shapes)
    print("Different Pancromatic patch sizes:", unique_p_shapes)

# Define patch size
patch_height = 214
patch_width = 194
get_patches_no_overlaped(LR_img_left, HR_img_left, P_img_left, patch_height, patch_width, "no_overlaped_left_0.45")
get_patches_no_overlaped(LR_img_right, HR_img_right, P_img_right, patch_height, patch_width, "no_overlaped_right_0.55")
get_patches_overlaped(LR_img_left, HR_img_left, P_img_left, patch_height, patch_width, "overlaped_left_0.45")
get_patches_overlaped(LR_img_right, HR_img_right, P_img_right, patch_height, patch_width, "overlaped_right_0.55")


