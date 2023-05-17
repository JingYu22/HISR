# -*- encoding: utf-8 -*-
from torch import optim
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from module.f_high import *

#换成spe attn
#RGB的高频信息（Gassian）和cat结果做cross attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Attention(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        # self.qkv_C = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.kv_C = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_C = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        # self.qkv_dwconv_C = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False)
        self.kv_dwconv_C = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.q_dwconv_C = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        self.proj_C = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, fhigh):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        #和fuseformer attn类似
        B_, N, C = x.shape
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))   

        # spectral attention
        # print(fhigh.shape) torch.Size([1, 3, 64, 64])
        # print(x.shape) torch.Size([64, 64, 32])
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh) #torch.Size([64, 32, 8, 8])
        # print(x.shape)
        # qkv_c = self.qkv_dwconv_C(self.qkv_C(x))
        # q_c,k_c,v_c = qkv_c.chunk(3, dim=1) 
        q_c = self.q_dwconv_C(self.q_C(fhigh))
        k_c = self.kv_dwconv_C(self.kv_C(x))
        v_c = self.kv_dwconv_C(self.kv_C(x))
        # print(q_c.shape)
        # print(k_c.shape, v_c.shape)
        # print("head",self.num_heads)
        q_c = rearrange(q_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_c = rearrange(k_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_c = rearrange(v_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_c = torch.nn.functional.normalize(q_c, dim=-1)#torch.Size([1, 3, 1, 4096])
        k_c = torch.nn.functional.normalize(k_c, dim=-1)#torch.Size([64, 3, 11, 64])
        
        
        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.temperature
        attn_c = attn_c.softmax(dim=-1)
        x2 = (attn_c @ v_c)
        x2 = rearrange(x2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=hh, w=hh)
        x2 = self.proj_C(x2)
        x2 = rearrange(x2, ' b c h w -> b (h w) c', h = hh, w = hh)

        # x3 = x1 + x2  
        return x2

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Window_Attention(nn.Module):
    r""" PSRT Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim=32, input_resolution=16, num_heads=8, window_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=4, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.window_size <= input_resolution, "input_resolution should be larger than window_size"

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self,H, W, x, fhigh):

        x = rearrange(x, 'B C H W -> B (H W) C')
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # attention
        attn_windows = self.attn(x_windows,fhigh)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        
        return x

def Win_Shuffle(x, win_size):
    """
    :param x: B C H W
    :param win_size:
    :return: y: B C H W
    """
    B, C, H, W = x.shape
    dilation = win_size // 2
    resolution = H
    assert resolution % win_size == 0, 'resolution of input should be divisible'
    assert win_size % 2 == 0, 'win_size should be the multiple of two'
    "input size BxCxHxW"
    "shuffle"

    N1 = H // dilation
    N2 = W // dilation
    x = rearrange(x, 'B C H W -> B H W C')
    x = window_partition(x, dilation)  # BN x d x d x c
    x = x.reshape(-1, N1, N2, C * dilation ** 2)  # B x n x n x d2c
    xt = torch.zeros_like(x)
    x0 = x[:, 0::2, 0::2, :]  # B n/2 n/2 d2c
    x1 = x[:, 0::2, 1::2, :]  # B n/2 n/2 d2c
    x2 = x[:, 1::2, 0::2, :]  # B n/2 n/2 d2c
    x3 = x[:, 1::2, 1::2, :]  # B n/2 n/2 d2c

    xt[:, 0:N1 // 2, 0:N2 // 2, :] = x0  # B n/2 n/2 d2c
    xt[:, 0:N1 // 2, N2 // 2:N2, :] = x1  # B n/2 n/2 d2c
    xt[:, N1 // 2:N1, 0:N2 // 2, :] = x2  # B n/2 n/2 d2c
    xt[:, N1 // 2:N1, N2 // 2:N2, :] = x3  # B n/2 n/2 d2c
    xt = xt.reshape(-1, dilation, dilation, C)
    xt = window_reverse(xt, dilation, H, W)
    xt = rearrange(xt, 'B H W C -> B C H W')

    return xt

def Win_Reshuffle(x, win_size):
    """
        :param x: B C H W
        :param win_size:
        :return: y: B C H W
        """
    B, C, H, W = x.shape
    dilation = win_size // 2
    N1 = H // dilation
    N2 = W // dilation
    assert H % win_size == 0, 'resolution of input should be divisible'
    assert win_size % 2 == 0, 'win_size should be the multiple of two'

    x = rearrange(x, 'B C H W -> B H W C')
    x = window_partition(x, dilation)  # BN x d x d x c
    x = x.reshape(-1, N1, N2, C * dilation ** 2)  # B x n x n x d2c
    xt = torch.zeros_like(x)
    xt[:, 0::2, 0::2, :] = x[:, 0:N1// 2, 0:N2 // 2, :]  # B n/2 n/2 d2c
    xt[:, 0::2, 1::2, :] = x[:, 0:N1 // 2, N2 // 2:N2, :]  # B n/2 n/2 d2c
    xt[:, 1::2, 0::2, :] = x[:, N1 // 2:N1, 0:N2 // 2, :]  # B n/2 n/2 d2c
    xt[:, 1::2, 1::2, :] = x[:, N1 // 2:N1, N2 // 2:N2, :]  # B n/2 n/2 d2c
    xt = xt.reshape(-1, dilation, dilation, C)
    xt = window_reverse(xt, dilation, H, W)
    xt = rearrange(xt, 'B H W C -> B C H W')

    return xt

class W2W_Block(nn.Module):
    def __init__(self, img_size=64, in_chans=32, head=8, win_size=4):
        """
        input: B x F x H x W
        :param img_size: size of image
        :param in_chans: feature of image
        :param embed_dim:
        :param token_dim:
        """
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_chans
        self.win_size = win_size
        self.WA1 = Window_Attention(dim=self.in_channels, input_resolution=img_size, num_heads=head,
                                    window_size=win_size)
        self.WA2 = Window_Attention(dim=self.in_channels, input_resolution=img_size, num_heads=head,
                                    window_size=win_size)
        self.WA3 = Window_Attention(dim=self.in_channels, input_resolution=img_size, num_heads=head,
                                    window_size=win_size)


    def forward(self, H, W, x, fhigh):
        # window_attention1
        shortcut = x
        x = self.WA1(H, W, x, fhigh)

        # shuffle
        x = Win_Shuffle(x, self.win_size)

        # window_attention2
        x = self.WA2(H, W, x, fhigh)

        # reshuffle
        x = Win_Reshuffle(x, self.win_size)

        # window_attention3
        x = self.WA3(H, W, x, fhigh)

        x = x + shortcut

        return x

class Pyramid_Block(nn.Module):
    def __init__(self, num=3, img_size=64, in_chans=32, head=8, win_size=8):
        """
        input: B x H x W x F
        :param img_size: size of image
        :param in_chans: feature of image
        :param num: num of layer
        """
        super().__init__()
        self.num_layers = num
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = W2W_Block(img_size=img_size, in_chans=in_chans, head=head, win_size=win_size//(2**i_layer))
            self.layers.append(layer)

    def forward(self, H, W, x, fhigh):
        for layer in self.layers:
            x = layer(H, W, x, fhigh)
        return x

class Block(nn.Module):
    def __init__(self, out_num, inside_num, img_size, in_chans, embed_dim, head, win_size):
        super().__init__()
        self.num_layers = out_num
        self.layers = nn.ModuleList()
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        for i_layer in range(self.num_layers):
            layer = Pyramid_Block(num=inside_num, img_size=img_size, in_chans=embed_dim, head=head, win_size=win_size)
            self.layers.append(layer)

    def forward(self, H, W, x, fhigh):
        x = self.conv(x)
        for layer in self.layers:
            x = layer(H, W, x, fhigh)
        return x

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def init_w(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # torch.nn.init.uniform_(m.weight, a=0, b=1)
            elif isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

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
    
class PSRTnet(nn.Module):
    def __init__(self, args):
        super(PSRTnet, self).__init__()
        self.args = args
        self.img_size = 64
        self.in_channels = 31
        self.embed = 33
        self.conv = nn.Sequential(
            nn.Conv2d(self.embed, self.in_channels, 3, 1, 1), nn.LeakyReLU(0.2, True)
        )
        # self.w = Block(num=2, img_size=self.img_size, in_chans=34, embed_dim=32, head=8, win_size=2)
        self.w = Block(out_num=2, inside_num=3, img_size=self.img_size, in_chans=34, embed_dim=self.embed, head=3,
                       win_size=8)
        self.visual_corresponding_name = {}
        init_weights(self.conv)
        init_w(self.w)
        self.high = GaussionBlur(kernel_size=3,sigma=1.5,num_channels=3)

    def forward(self, rgb, lms):
        '''
        :param rgb:
        :param lms:
        :return:
        '''
        self.rgb = rgb
        self.lms = lms
        self.rgb_high = self.high(self.rgb)#torch.Size([16, 3, 64, 64])
        # self.rgb = self.rgb + self.rgb_high
        xt = torch.cat((self.lms, self.rgb), 1)  # Bx34X64x64
        _, _, H, W = xt.shape
        w_out = self.w(H, W, xt, self.rgb_high)
        self.result = self.conv(w_out) + self.lms
        # print(self.high(self.rgb).shape)#torch.Size([1, 3, 64, 64])
        # self.result = self.result + self.high(self.rgb)

        return self.result

    def name(self):
        return 'PSRT'

