import argparse
import glob
import numpy as np
import os
from scipy import ndimage
import tifffile
import utils
import pandas as pd
from tqdm import tqdm
import math

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="whole_image", type=str, help="'patches' or 'whole_image'")
parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
parser.add_argument('--folder_lq', type=str, default="all_HR_no_overlaped_left_0.45", help='input low-quality test image folder')
parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
parser.add_argument('--order', type=int, default=3, help='interpolation order')
parser.add_argument('--windows_size', type=int, default=8, help='window size')
parser.add_argument('--WI_path', default="datasets/whole_images/RGBNED.tif", type=str, help='Path to the whole image')
args = parser.parse_args()

def interpolation_patches(args):
    
    border=args.scale
    results_dic={}
    results_dic['PSNR']=[]
    #results_dic['SSIM']=[]
    images_list = sorted(glob.glob(os.path.join('patches',args.folder_gt, '*.tif')))
    img_ignored = []

    for idx, path in tqdm(enumerate(images_list), total=len(images_list)):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        
        #output = ndimage.zoom(np.squeeze(img_lq[...]), args.scale, order=args.order) # Extract the first channel and remove the last dimension
        output0 = ndimage.zoom(np.squeeze(img_lq[...,0]), args.scale, order=args.order) # Extract the first channel and remove the last dimension
        output1 = ndimage.zoom(np.squeeze(img_lq[...,1]), args.scale, order=args.order) # Extract the second channel and remove the last dimension
        output2 = ndimage.zoom(np.squeeze(img_lq[...,2]), args.scale, order=args.order) # Extract the third channel and remove the last dimension
        output3 = ndimage.zoom(np.squeeze(img_lq[...,3]), args.scale, order=args.order) # Extract the fourth channel and remove the last dimension
        output4 = ndimage.zoom(np.squeeze(img_lq[...,4]), args.scale, order=args.order) # Extract the fifth channel and remove the last dimension
        output5 = ndimage.zoom(np.squeeze(img_lq[...,5]), args.scale, order=args.order) # Extract the sixth channel and remove the last dimension"""
        
        output0 = np.expand_dims(output0, axis=2) # Add third dimensions  
        output1 = np.expand_dims(output1, axis=2) # Add third dimensions  
        output2 = np.expand_dims(output2, axis=2) # Add third dimensions
        output3 = np.expand_dims(output3, axis=2) # Add third dimensions  
        output4 = np.expand_dims(output4, axis=2) # Add third dimensions  
        output5 = np.expand_dims(output5, axis=2) # Add third dimensions 

        output = np.concatenate((output0, output1, output2, output3, output4, output5), axis=2) # Concatenate over the channels dimension
        
        # Evaluate psnr
        if img_gt is not None:
            #img_gt = (img_gt * 65535.0).round().astype(np.uint16)  # float32 to uint16            
            psnr = utils.calculate_psnr(output, img_gt, border=border)
            if not math.isinf(psnr):
                results_dic['PSNR'].append(psnr)
            else:
                img_ignored.append(imgname)
        
        # Create the 'results' folder if it doesn't exist
        order_lq = args.folder_lq.split("_")[-1]
        results_folder = f'results/interpolation_6channels_order{args.order}_{order_lq}' 
        os.makedirs(results_folder, exist_ok=True)

        # Save the concatenated array as a TIFF file
        output_file = os.path.join(results_folder, imgname + '.tif')
        tifffile.imsave(output_file, output)

    # Save metrics
    if img_gt is not None:
        psnr_mean=np.mean(results_dic['PSNR'])
        psnr_desv=np.std(results_dic['PSNR'])
        print('PSNR mean: ', psnr_mean, 'PSNR desv: ', psnr_desv)
        results_df=pd.DataFrame(results_dic)
        results_df.to_csv(f'results/interpolation_6channels_order{args.order}_{order_lq}.csv')
        print(img_ignored)

def interpolation_WI(args):

    # Read image
    img_lq = tifffile.imread(args.WI_path)

    #output = ndimage.zoom(np.squeeze(img_lq[...]), args.scale, order=args.order) # Extract the first channel and remove the last dimension
    output0 = ndimage.zoom(np.squeeze(img_lq[...,0]), args.scale, order=args.order) # Extract the first channel and remove the last dimension
    output1 = ndimage.zoom(np.squeeze(img_lq[...,1]), args.scale, order=args.order) # Extract the second channel and remove the last dimension
    output2 = ndimage.zoom(np.squeeze(img_lq[...,2]), args.scale, order=args.order) # Extract the third channel and remove the last dimension
    output3 = ndimage.zoom(np.squeeze(img_lq[...,3]), args.scale, order=args.order) # Extract the fourth channel and remove the last dimension
    output4 = ndimage.zoom(np.squeeze(img_lq[...,4]), args.scale, order=args.order) # Extract the fifth channel and remove the last dimension
    output5 = ndimage.zoom(np.squeeze(img_lq[...,5]), args.scale, order=args.order) # Extract the sixth channel and remove the last dimension"""
    
    output0 = np.expand_dims(output0, axis=2) # Add third dimensions  
    output1 = np.expand_dims(output1, axis=2) # Add third dimensions  
    output2 = np.expand_dims(output2, axis=2) # Add third dimensions
    output3 = np.expand_dims(output3, axis=2) # Add third dimensions  
    output4 = np.expand_dims(output4, axis=2) # Add third dimensions  
    output5 = np.expand_dims(output5, axis=2) # Add third dimensions 

    output = np.concatenate((output0, output1, output2, output3, output4, output5), axis=2) # Concatenate over the channels dimension
    
    # Create the 'results' folder if it doesn't exist
    os.makedirs(f'results' , exist_ok=True)

    # Save the concatenated array as a TIFF file
    output_file = os.path.join(f'results/interpolation_WI_order{args.order}.tif')
    tifffile.imsave(output_file, output)

def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ['classical_sr', 'lightweight_sr']:
        #img_gt = tifffile.imread(path).astype(np.float32) / 65535.
        img_gt = tifffile.imread(path).astype(np.uint16)
        img_gt = np.transpose(img_gt, (1, 2, 0))
        img_lq = tifffile.imread(f'patches/{args.folder_lq}/{imgname}{imgext}').astype(np.uint16)
        #img_lq = tifffile.imread(f'{args.folder_lq}/{imgname}{imgext}').astype(np.float32) / 65535.
        img_lq = np.transpose(img_lq, (1, 2, 0))

    # 003 real-world image sr (load lq image only)
    elif args.task in ['real_sr']:
        img_gt = None
        img_lq = tifffile.imread(path).astype(np.float32) / 65535.

    return imgname, img_lq, img_gt

def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    return folder, save_dir, border, window_size

if __name__ == '__main__':
    if args.mode == 'patches':
        interpolation_patches(args)
    else:
        interpolation_WI(args)