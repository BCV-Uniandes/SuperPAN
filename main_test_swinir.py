import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import tifffile

from models.network_swinir import SwinIR as net
from models.network_swinir_modified import SwinIR as net_modified
from utils import utils_image as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--folder_pan', type=str, default=None, help='input pancromatic test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    # Boolean argument for loading the baseline model or the pancromatic model
    parser.add_argument('--pan', action='store_true', help='use pancromatic model') 
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)
    
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    psnr = 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read images as tensors expected by model
        imgname, img_lq, img_gt, img_pan = get_image_pair(args, path) 
        # Send data to device (gpu or cpu)
        img_lq = img_lq.to(device)
        img_gt = img_gt.to(device)
        img_pan = img_pan.to(device)
        
        # inference
        with torch.no_grad():
            output = test(img_lq, img_pan, model, args, window_size)

        # tensor output to uint16
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = output.transpose(1, 2, 0)
        output = (output * 65535.0).round().astype(np.uint16)  # float32 to uint16
        # Save output image in tifffile format
        tifffile.imsave(f'{save_dir}/{imgname}.tif', output)
        # tensor img_gt to uint16
        if img_gt is not None:
            img_gt = img_gt.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img_gt.ndim == 3:
                img_gt = img_gt.transpose(1, 2, 0)
            img_gt = (img_gt * 65535.0).round().astype(np.uint16)  # float32 to uint16

            # evaluate psnr/ssim/psnr_b
            psnr = util.calculate_psnr(output, img_gt, border=border)
            test_results['psnr'].append(psnr)

            print('Testing {:d} {:20s} - PSNR: {:.2f} dB'.format(idx, imgname, psnr))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        print('\n{} \n-- Average PSNR: {:.2f} dB'.format(save_dir, ave_psnr))


def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        if args.pan: # use pancromatic model
            model = net_modified(upscale=args.scale, in_chans=6, img_size=args.training_patch_size, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        else: # use baseline model
            model = net(upscale=args.scale, in_chans=6, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    return folder, save_dir, border, window_size

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def single2tensor3(img):
    img = np.ascontiguousarray(img)  # Convert to contiguous NumPy array
    if img.ndim == 2:  # Image has 1 channel (Pancromatic)
        img = img[np.newaxis, :, :]  # Add a new axis for channel dimension
    elif img.ndim == 3:  # Image has multiple channels
        img = np.transpose(img, (2, 0, 1))  # Permute dimensions for PyTorch format
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return torch.from_numpy(img).float()  # Convert to PyTorch tensor and set data type to float

def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs) adapted for tifffile
    if args.task in ['classical_sr', 'lightweight_sr']:
        # High-quality (HQ) image
        img_gt = tifffile.imread(path).astype(np.uint16)
        img_gt = np.transpose(img_gt, (1, 2, 0))
        img_gt = img_gt.astype(np.float32) / 65535.
        # Modcrop img_gt to be a multiple of scale
        img_gt = modcrop(img_gt, args.scale)
        # Single image to tensor
        img_gt = single2tensor3(img_gt)
        # Add batch dimension: 1xCxHxW
        img_gt = img_gt.unsqueeze(0)

        # Low-quality (LQ) image
        img_lq = tifffile.imread(f'{args.folder_lq}/{imgname}{imgext}').astype(np.uint16)
        img_lq = np.transpose(img_lq, (1, 2, 0))
        img_lq = img_lq.astype(np.float32) / 65535.
        # Single image to tensor
        img_lq = single2tensor3(img_lq)
        # Add batch dimension: 1xCxHxW
        img_lq = img_lq.unsqueeze(0)
        
        # Pancromatic (pan) image
        img_pan = tifffile.imread(f'{args.folder_pan}/{imgname}{imgext}').astype(np.uint16) # No transpose because it is a single channel image
        img_pan = img_pan.astype(np.float32) / 65535.
        # Modcrop img_pan to be a multiple of scale
        img_pan = modcrop(img_pan, args.scale)
        # Single image to tensor
        img_pan = single2tensor3(img_pan)
        # Add batch dimension: 1xCxHxW
        img_pan = img_pan.unsqueeze(0)

    return imgname, img_lq, img_gt, img_pan


def test(img_lq, img_pan, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq, img_pan)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
