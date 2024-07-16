import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse
import glob

# Create argparse object
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=2, help='fold number')
parser.add_argument('--number_of_runs', type=int, default=8, help='number of runs')
args = parser.parse_args()

# Obtain folders in current directory with the fold number argument
folders = [f for f in os.listdir('.') if os.path.isdir(f) and (f'fold{args.fold}') in f]
num_images = len(folders)

# Change folders order
our = folders[0]
folders[0]=folders[2]
folders[2]=folders[1]
folders[1]=our

# Add numpy seed
np.random.seed(0)

for i in range(args.number_of_runs):

    # Obtain 1 file randomly (the files have the same name in each folder)
    file = np.random.choice(os.listdir(folders[0]))

    # Read the random chosen file in tifffile format as uint16 for each folder
    images = [tf.imread(os.path.join(folder, file)).astype('uint16') for folder in folders]

    # Transpose Ground Truth and Input
    images[0] = np.transpose(images[0], (1, 2, 0))
    images[2] = np.transpose(images[2], (1, 2, 0))
    
    # Convert the data range to [0, 255] for visualization
    images = [(image / np.iinfo('uint16').max * 255).astype('uint8') for image in images]

    # Consider only the first 3 channels (RGB) for visualization
    images = [image[:,:,:3] for image in images]

    # Convert RGB to HSV
    images = [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in images]

    # Equalize the value channel
    equalized_channels = [cv2.equalizeHist(image[:,:,2]) for image in images]

    # Merge the equalized value channel with the original hue and saturation channels
    images = [cv2.merge((h, s, v)) for h, s, v in zip([image[:,:,0] for image in images], [image[:,:,1] for image in images], equalized_channels)]

    # Convert back to RGB
    images = [cv2.cvtColor(image, cv2.COLOR_HSV2RGB) for image in images]

    # Sizes of the images
    sizes = [img.shape for img in images]

    # Plot the whole image in a 1xnum_images subplot
    fig, axs = plt.subplots(1, num_images, figsize=(30, 10))  # Increase the figure size
    # Remove the x and y tick labels for all subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    # Iterate over the images and their corresponding titles and sizes
    for ax, img, title, size in zip(axs, images, folders, sizes):
        ax.imshow(img)
        ax.set_title(f'{title}')  # Include the size in the title
    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()
    # Save the plot in the folder "plots/fold{args.fold}"
    os.makedirs(os.path.join("plots",f"fold{args.fold}"), exist_ok=True)
    # Name the plot with the number of images in the folder in which the images are stored
    num_images_saved = len(os.listdir(os.path.join("plots", f"fold{args.fold}")))//2 # Divide by 2 because there are 2 plots saved per run
    fig.savefig(os.path.join("plots", f"fold{args.fold}",f"result_{num_images_saved+1}.png"))

    # Plot a zoomed version of the image in a 1xnum_images subplot
    fig, axs = plt.subplots(1, num_images, figsize=(30, 10))  # Increase the figure size
    # Remove the x and y tick labels for all subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    # Iterate over the images and their corresponding titles and sizes
    for ax, img, title, size in zip(axs, images, folders, sizes):
        ax.imshow(img)
        ax.set_title(f'{title}\n{size[0]}x{size[1]}')  # Include the size in the title
        # Zoom in the image
        ax.set_xlim([size[1]//2 - 100, size[1]//2 + 100])
        ax.set_ylim([size[0]//2 - 100, size[0]//2 + 100])
    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()
    # Save the plot
    fig.savefig(os.path.join("plots", f"fold{args.fold}",f"result_zoomed_{num_images_saved+1}.png"))

