import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import math

def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/4*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt-img_fus)**2)
    img_max = np.max(img_tgt)
    # img_max = 1
    psnr = 10*np.log10(img_max**2/mse)

    return psnr

# def calc_psnr(img1, img2):
#     mse_sum  = (img1  - img2 )**2
#     mse_loss = mse_sum.mean(2).mean(2) 
#     mse = mse_sum.mean()                     #.pow(2).mean()
#     if mse < 1.0e-10:
#         return 100
#     PIXEL_MAX = 1
#     # print(mse)
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse

def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    A = np.sqrt(np.sum(img_tgt**2, axis=0))
    B = np.sqrt(np.sum(img_fus**2, axis=0))
    AB = np.sum(img_tgt*img_fus, axis=0)

    sam = AB/(A*B)
    sam = np.arccos(sam)
    sam = np.mean(sam)*180/3.1415926535

    return sam

# patch_size = 256

def calc_ssim(im1,im2,batch_size,patch_size): 
    
    if batch_size > 1:
        im1 = np.reshape(im1, (batch_size,patch_size,patch_size,31))
        im2 = np.reshape(im2, (batch_size,patch_size,patch_size,31))
        n = im1.shape[3]
        ms_ssim = 0.0
        for i in range(n):
            single_ssim = compare_ssim(im1[:,:,:,i], im2[:,:,:,i])
            ms_ssim += single_ssim
    else:
        im1 = np.reshape(im1, (patch_size,patch_size,31))
        im2 = np.reshape(im2, (patch_size,patch_size,31))
        n = im1.shape[2]
        ms_ssim = 0.0
        for i in range(n):
            single_ssim = compare_ssim(im1[:,:,i], im2[:,:,i])
            ms_ssim += single_ssim
    return ms_ssim/n

