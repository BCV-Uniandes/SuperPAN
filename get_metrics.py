from torchmetrics.image import ErrorRelativeGlobalDimensionlessSynthesis, SpectralAngleMapper
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pathlib import Path
from utils import calculate_psnr
import numpy as np
import argparse
import tifffile
import torch
import os
import cv2
import glob
from scipy.io import loadmat

def read_tif_image(file_path):
    # Use tifffile to read the image data
    data = tifffile.imread(file_path).astype(np.float32)
    return data

def create_batch_tensor(folder_path, task, preds):
    
    if task == 'lambdaPNN' and preds:
        mat_files = sorted(glob.glob(os.path.join(folder_path, "*.mat")))
        tensors = [torch.tensor(loadmat(file_path)["I_MS"][:,:,:6].astype(np.float32)) for file_path in mat_files]
        batch_tensor = torch.stack(tensors)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)
    else:
        # Initialize a transform to convert numpy arrays to PyTorch tensors
        to_tensor = ToTensor()
        # List all .tif files in the folder
        tif_files = sorted(Path(folder_path).glob('*.tif'))
        # Read and convert each image to a tensor
        tensors = [to_tensor(read_tif_image(str(file_path))) for file_path in tif_files]
        # Stack all image tensors to create a batch
        batch_tensor = torch.stack(tensors)

    return batch_tensor

def visualize_batch(pred_tensor, target_tensor):

    os.makedirs('results/batch_visualizations', exist_ok=True)

    for i in range(pred_tensor.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Ajusta el tamaño según sea necesario

        pred_img = pred_tensor[i][:3, :, :].permute(1, 2, 0).cpu().numpy()
        target_img = target_tensor[i][:3, :, :].permute(1, 2, 0).cpu().numpy()

        # Convert RGB to HSV
        pred_img = (pred_img * 255).astype(np.uint8)
        target_img = (target_img * 255).astype(np.uint8)
        pred_img_hsv = cv2.cvtColor(pred_img, cv2.COLOR_RGB2HSV)
        target_img_hsv = cv2.cvtColor(target_img, cv2.COLOR_RGB2HSV)

        # Equalize the value channel
        pred_img_hsv[:, :, 2] = cv2.equalizeHist(pred_img_hsv[:, :, 2])
        target_img_hsv[:, :, 2] = cv2.equalizeHist(target_img_hsv[:, :, 2])

        # No need to merge channels manually as the operation is done in-place

        # Convert back to RGB if needed for further processing or visualization
        pred_img_equalized_rgb = cv2.cvtColor(pred_img_hsv, cv2.COLOR_HSV2RGB)
        target_img_equalized_rgb = cv2.cvtColor(target_img_hsv, cv2.COLOR_HSV2RGB)

        # Convert back to RGB
        pred_img = cv2.cvtColor(pred_img_equalized_rgb, cv2.COLOR_HSV2RGB)
        target_img = cv2.cvtColor(target_img_equalized_rgb, cv2.COLOR_HSV2RGB)

        # Visualizar la imagen predicha
        ax[0].imshow(pred_img)
        ax[0].set_title('Prediction')
        ax[0].axis('off')

        # Visualizar la imagen objetivo
        ax[1].imshow(target_img)
        ax[1].set_title('Target')
        ax[1].axis('off')

        plt.savefig(f'results/batch_visualizations/{i}.png')
        plt.close(fig) 

# Create argparser
parser = argparse.ArgumentParser(description='Get ERGAS metric for a set of images')
parser.add_argument('--task', type=str, default="interpolation", help='Task to be evaluated [interpolation, hyperes, pansharpening]') 
args = parser.parse_args()

# Path to target folder
# fold1 
target_folder1_path = 'patches/all_HR_no_overlaped_right_0.55'
# fold2
target_folder2_path = 'patches/all_HR_no_overlaped_left_0.45'

# Create the batch tensors for targets
target1_tensor = create_batch_tensor(target_folder1_path, args.task, False)
target2_tensor = create_batch_tensor(target_folder2_path, args.task, False)

# Rearrange the dimensions of target tensor according to the preds tensor order
target1_tensor = target1_tensor.permute(0, 2, 3, 1)
target2_tensor = target2_tensor.permute(0, 2, 3, 1)

# Path to preds folder
if args.task == 'interpolation':
    # Interpolation fold1
    preds_folder1_path = 'results/interpolation_6channels_order3_0.55'
    # Interpolation fold2
    preds_folder2_path = 'results/interpolation_6channels_order3_0.45'

elif args.task == 'hyperes_OP':
    # HypeResr fold1
    preds_folder1_path = "/media/SSD3/idchacon/MACAW/KAIR/results/swinir_srx4_classical_patches_overlapping_pancromatic_fold1_deadlinedifferent-sweep-1_x4"
    # HypeResr fold2
    preds_folder2_path = "/media/SSD3/idchacon/MACAW/KAIR/results/swinir_srx4_classical_patches_overlapping_pancromatic_fold2_deadlinebumbling-sweep-1_x4"

elif args.task == 'hyperes_NOP':
    # HypeResr fold1
    preds_folder1_path = "/media/SSD3/idchacon/MACAW/KAIR/results/swinir_srx4_classical_patches_no_overlapping_pancromatic_fold1_deadlineethereal-sweep-1_x4"
    # HypeResr fold2
    preds_folder2_path = "/media/SSD3/idchacon/MACAW/KAIR/results/swinir_srx4_classical_patches_no_overlapping_pancromatic_fold2_deadlinefallen-sweep-1_x4"

elif args.task == 'pan_Brovey':
    # Pansharpening fold1
    preds_folder1_path = "/media/user_home1/lfvargas/macaw_cvpr/pansharpened/BroveyRGB/no_overlaped_right_0.55"
    # Pansharpening fold2
    preds_folder2_path = "/media/user_home1/lfvargas/macaw_cvpr/pansharpened/BroveyRGB/no_overlaped_left_0.45"

elif args.task == 'pan_PCA':
    # Pansharpening fold1
    preds_folder1_path = "/media/user_home1/lfvargas/macaw_cvpr/pansharpened/PCARGB/no_overlaped_right_0.55"
    # Pansharpening fold2
    preds_folder2_path = "/media/user_home1/lfvargas/macaw_cvpr/pansharpened/PCARGB/no_overlaped_left_0.45"

elif args.task == 'lambdaPNN':
    # Pansharpening fold1
    preds_folder1_path = "/media/SSD3/idchacon/MACAW/KAIR/results/results_LambdaPNN/all_Predicted_no_overlaped_left_0.45/WV2/L-PNN"
    # Pansharpening fold2
    preds_folder2_path = "/media/SSD3/idchacon/MACAW/KAIR/results/results_LambdaPNN/all_Predicted_no_overlaped_left_0.45/WV2/L-PNN"

if 'pan' in args.task:
    target1_tensor = target1_tensor[:, :3, :, :]
    target2_tensor = target2_tensor[:, :3, :, :]
    preds1_tensor = create_batch_tensor(preds_folder1_path, args.task, True)[:, :3, :, :]
    preds2_tensor = create_batch_tensor(preds_folder2_path, args.task, True)[:, :3, :, :]
else:
    preds1_tensor = create_batch_tensor(preds_folder1_path, args.task, True)
    preds2_tensor = create_batch_tensor(preds_folder2_path, args.task, True)

print(f'Preds folder1 Tensor Shape: {preds1_tensor.shape}')
print(f'Preds folder2 Tensor Shape: {preds2_tensor.shape}')
print(f'Target folder1 Tensor Shape: {target1_tensor.shape}')
print(f'Target folder2 Tensor Shape: {target2_tensor.shape}')

# Visualize the batch
#visualize_batch(preds_tensor, target_tensor)

# ERGAS METRIC
ratio = 4  
ergas = ErrorRelativeGlobalDimensionlessSynthesis(ratio=ratio, reduction="elementwise_mean")
ergas_folder1 = ergas(preds1_tensor, target1_tensor)
ergas_folder2 = ergas(preds2_tensor, target2_tensor)
print(args.task)
print(f'ERGAS Folder1: {ergas_folder1.item()}')
print(f'ERGAS Folder2: {ergas_folder2.item()}')

# SAM METRIC
sam = SpectralAngleMapper()
sam_folder1 = sam(preds1_tensor, target1_tensor)
sam_folder2 = sam(preds2_tensor, target2_tensor)
print(args.task)
print(f'SAM Folder1: {sam_folder1.item()}')
print(f'SAM Folder2: {sam_folder2.item()}')



