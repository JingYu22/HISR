import numpy as np
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
import math
from scipy.io import loadmat,savemat
import torch

patch_size = 512

def calc_psnr(img1, img2):
    mse_sum  = (img1  - img2 )**2
    mse_loss = mse_sum.mean(2).mean(2) 
    mse = mse_sum.mean()                     #.pow(2).mean()
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # print(mse)
    return mse_loss, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
def calc_sam(im1, im2):
    im1 = np.reshape(im1,(patch_size*patch_size,31))
    im2 = np.reshape(im2,(patch_size*patch_size,31))
    mole = np.sum(np.multiply(im1, im2), axis=1) 
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)
    sam = np.rad2deg(np.arccos(((mole+10e-8)/(deno+10e-8)).clip(-1,1)))
    return np.mean(sam)

def calc_ssim(im1,im2): 
    im1 = np.reshape(im1, (patch_size,patch_size,31))
    im2 = np.reshape(im2, (patch_size,patch_size,31))
    n = im1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        single_ssim = compare_ssim(im1[:,:,i], im2[:,:,i])
        ms_ssim += single_ssim
    return ms_ssim/n

def calc_ergas(mse, out):
    out = np.reshape(out, (patch_size*patch_size,31))
    out_mean = np.mean(out, axis=0)
    mse = np.reshape(mse, (31, 1))
    out_mean = np.reshape(out_mean, (31, 1))
    ergas = 100/8*np.sqrt(np.mean(mse/out_mean**2))                    
    return ergas