import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
import os
from glob import glob
import os.path as osp
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from einops import rearrange, reduce, repeat


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(1. / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, channel_axis=2, data_range=1.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', help='Path to the ground truth directory')
    parser.add_argument('--render', help='Path to the rendering directory')
    args = parser.parse_args()
    lpips_model = LPIPS(net='alex').cuda()

    psnrs = []
    ssims = []
    lpipss = []

    gt_fns = glob(osp.join(str(args.gt), '*.png'))
    for gt_fn in tqdm(gt_fns[:10]):
        fn = osp.basename(gt_fn)
        render_fn = osp.join(args.render, fn)
        gt_img = plt.imread(gt_fn)
        render_img = plt.imread(render_fn)


        psnr = calculate_psnr(gt_img, render_img)
        ssim = calculate_ssim(gt_img, render_img)
        lpips = lpips_model(torch.tensor(rearrange(gt_img, 'h w c -> 1 c h w')).cuda(), 
                            torch.tensor(rearrange(render_img, 'h w c -> 1 c h w')).cuda(),).item()

        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lpips)

    print(f'PSNR: {np.mean(psnrs):.4f}')
    print(f'SSIM: {np.mean(ssims):.4f}')
    print(f'LPIPS: {np.mean(lpipss):.4f}')

if __name__ == '__main__':
    main()