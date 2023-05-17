import torch
import torch.nn as nn
import math
import einops
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import numpy as np

class GaussionBlur(nn.Module):
    def __init__(self,kernel_size=3,sigma=1.5,num_channels=3):
        super().__init__()
        # Create Gaussian kernel
        # 法1
        x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size), torch.linspace(-1, 1, kernel_size))
        gaussian_kernel = torch.exp(-(x**2 + y**2) / (2*sigma**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # 法2
        # x_coord = torch.arange(kernel_size)
        # x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        # y_grid = x_grid.t()
        # xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        # mean = (kernel_size - 1)/2.
        # variance = sigma**2.
        # gaussian_kernel = (1./(2.*math.pi*variance)) *\
        #                 torch.exp(
        #                     -torch.sum((xy_grid - mean)**2., dim=-1) /\
        #                     (2*variance)
        #                 )
        # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        # gaussian_kernel = torch.tensor(gaussian_kernel).float().unsqueeze(0).unsqueeze(0)

        # Expand the kernel to the desired number of channels
        
        gaussian_kernel = gaussian_kernel.expand(num_channels, 3, kernel_size, kernel_size)
        # Define the convolutional layer
        # self.conv = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        # Assign the Gaussian kernel as the weight of the convolutional layer
        self.conv.weight.data = gaussian_kernel
        self.conv.weight.requires_grad = False
        
    def forward(self,x):
        x_low = self.conv(x)
        x_high = x-x_low
        return x_high
    
class ReconBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv_s = nn.Sequential(nn.Conv2d(256,256,3,1,1),
        #                             nn.LeakyReLU(),
        #                             nn.Conv2d(256,256,3,1,1),
        #                             nn.LeakyReLU())
        self.conv_s = nn.Sequential(nn.Conv2d(768,768,3,1,1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(768,768,3,1,1),
                                    nn.LeakyReLU())
        self.up_s = nn.PixelShuffle(2)

        # self.conv_d = nn.Sequential(nn.Conv2d(64,64,3,1,1),
        #                             nn.LeakyReLU(),
        #                             nn.Conv2d(64,64,3,1,1),
        #                             nn.LeakyReLU())
        self.conv_d = nn.Sequential(nn.Conv2d(192,192,3,1,1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(192,192,3,1,1),
                                    nn.LeakyReLU())
        self.up_d = nn.PixelShuffle(8)

    def forward(self,x):
        # print(x.shape)
        x = self.conv_s(x) + x
        x = self.up_s(x)
        x = self.conv_d(x) + x
        x_out = self.up_d(x)

        return x_out
    
class TRANSCT(nn.Module):
    def __init__(self,trans_depth=3):
        super().__init__()
        # num_slice = 1
        num_slice = 3

        self.gaussion = GaussionBlur(kernel_size=3,sigma=1,num_channels=num_slice)

        # self.head_low = nn.Sequential(nn.Conv2d(num_slice,16,3,2,1),
        #                               nn.LeakyReLU(),
        #                               nn.Conv2d(16,32,3,2,1),
        #                               nn.LeakyReLU())
        down_scale=16
        self.head_high = nn.Sequential(nn.PixelUnshuffle(down_scale),
                                      nn.Conv2d(num_slice*(down_scale**2),num_slice*(down_scale**2),3,1,1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(num_slice*(down_scale**2),num_slice*(down_scale**2),3,1,1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(num_slice*(down_scale**2),num_slice*(down_scale**2),3,1,1),
                                      nn.LeakyReLU())

        # self.freq_low = nn.Sequential(nn.Conv2d(32,64,3,2,1),
        #                               nn.LeakyReLU(),
        #                               nn.Conv2d(64,128,3,2,1),
        #                               nn.LeakyReLU(),
        #                               nn.Conv2d(128,256,3,2,1),
        #                               nn.LeakyReLU())
        # self.low1 = nn.Sequential(nn.Conv2d(32,64,3,2,1),
        #                               nn.LeakyReLU())                                     
        # self.low2 = nn.Sequential(nn.Conv2d(64,256,3,2,1),
        #                               nn.LeakyReLU())   

        # self.trans = nn.ModuleList([TransBlock() for _ in range(trans_depth)])
        # self.encoder = nn.ModuleList([Encoder() for _ in range(trans_depth)])
        # self.decoder = nn.ModuleList([Decoder() for _ in range(trans_depth)])

        self.recon = ReconBlock()

    def forward(self,x):
        # x[bz,c,h,w]

        x_low,x_high = self.gaussion(x)

        # high
        x_head_high = self.head_high(x_high)
        B,C,H,W = x_head_high.shape
        f_high = x_head_high
       
        x_out = self.recon(f_high)
        return x_out