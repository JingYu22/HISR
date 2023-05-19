import numpy as np
from metrics import calc_psnr, calc_ergas, calc_sam, calc_ssim
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import scipy.io as sio
import h5py
from os.path import exists, join, basename
import torch.utils.data as data
from model.model_0515 import *
import torch.nn.functional as F
# from cal_ssim import SSIM

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

name = '/cave11-0515.mat'

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')
        print(dataset.keys())
        self.GT = dataset.get("GT")
        self.UP = dataset.get("HSI_up")
        self.LRHSI = dataset.get("HSI_up")
        self.RGB = dataset.get("RGB")

    #####必要函数
    def __getitem__(self, index):
        input_pan = torch.from_numpy(self.RGB[index, :, :, :]).float()
        input_lr = torch.from_numpy(self.LRHSI[index, :, :, :]).float()
        input_lr_u = torch.from_numpy(self.UP[index, :, :, :]).float()
        target = torch.from_numpy(self.GT[index, :, :, :]).float()

        return input_pan, input_lr, input_lr_u, target
        #####必要函数

    def __len__(self):
        return self.GT.shape[0]

def get_test_set(root_dir):
    train_dir = join(root_dir, "test_cavepatches256-2.h5")
    # train_dir = join(root_dir, "test_cave(with_up)x4.h5")
    
    return DatasetFromHdf5(train_dir)

from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--dataset', type=str, default='/data/yj/PSRT/cave_x4')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--n_bands', type=int, default=31)
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--model_path', type=str,
                    default='/home/ubuntu/106-48t/personal_data/yj/local/checkpoints/PSRT_cave_x4_202305171102/model_epoch_2000.pth.tar',
                    help='path for trained encoder')
opt = parser.parse_args()

test_set = get_test_set(opt.dataset)
test_data_loader = DataLoader(dataset=test_set,  batch_size=opt.testBatchSize, shuffle=False)


def cleanup_state_dict(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if "module" in k:
            new_name = k[7:]
        else:
            new_name = k
        new_state[new_name] = v
    return new_state

def test(test_data_loader):
    model = PSRTnet(opt).cuda()

    checkpoint = torch.load(opt.model_path)
    # print(opt.model_path)
    # model.load_state_dict(checkpoint["model"].state_dict())

    # model = torch.load(opt.model_path)

    # if you want to use the pretrained checkpoints, use the blow code instead.
    # state_dict = checkpoint['model']
    # dict = {}
    # for module in state_dict.items():
    #     k, v = module
    #     if 'model' in k:
    #         k = k.strip('model.')
    #     dict[k] = v
    # checkpoint['state_dict'] = dict
    # model.load_state_dict(checkpoint['state_dict'])

    model.load_state_dict(cleanup_state_dict(checkpoint["model"].state_dict()))
    


    model.eval()
    output = np.zeros((44, opt.image_size, opt.image_size, opt.n_bands))
    
    # calc_ssim = SSIM(size_average=True)
    psnr_list = []
    sam_list = []
    ergas_list = []
    ssim_list = []

    for index, batch in enumerate(test_data_loader):
        input_rgb, _, input_lr_u, ref = Variable(batch[0]).cuda(), Variable(batch[1],).cuda(), Variable(batch[2]).cuda(), Variable(batch[3]).cuda()
        
        # input_lr = F.interpolate(input_lr_u, scale_factor=1/4, mode='bicubic')
        # ref = ref.cuda()

        out = model(input_rgb, input_lr_u)
        # out = model(input_rgb, input_lr)

        ref = ref.detach().cpu().numpy()#(1, 31, 256, 256)
        out1 = out.detach().cpu().numpy()#(1, 31, 256, 256)
        # print(out.shape)

        psnr = calc_psnr(ref, out1)
        sam = calc_sam(ref, out1)
        ergas = calc_ergas(ref, out1)
        ssim = calc_ssim(ref, out1, batch_size=ref.shape[0],patch_size=ref.shape[2])

        psnr_list.append(psnr)
        sam_list.append(sam)
        ergas_list.append(ergas)
        ssim_list.append(ssim)

        # rmse = calc_rmse(ref, out)
        # ergas = calc_ergas(ref, out)
        # sam = calc_sam(ref, out)
        # print('RMSE:   {:.4f};'.format(rmse))
        # print('PSNR:   {:.4f};'.format(psnr))
        # print('ERGAS:   {:.4f};'.format(ergas))
        # print('SAM:   {:.4f}.'.format(sam))

        output[index, :, :, :] = out.permute(0, 2, 3, 1).cpu().detach().numpy()
    print('PSNR:   {:.4f};'.format(np.array(psnr_list).mean()))
    print('SAM:   {:.4f};'.format(np.array(sam_list).mean()))
    print('ERGAS:   {:.4f};'.format(np.array(ergas_list).mean()))
    print('SSIM:   {:.4f};'.format(np.array(ssim_list).mean()))

    sio.savemat('./results' + name, {'output': output})



#
# if not os.path.exists(image_path):
#     os.makedirs(image_path)


test(test_data_loader)

file_path = '/home/ubuntu/106-48t/personal_data/yj/local/results' + name
PATH = '/home/ubuntu/106-48t/personal_data/yj/local/merge'
data_dic = sio.loadmat(file_path)
result = data_dic['output']
result = rearrange(result, '(b h1 w1) h w c  -> b (h1 h) (w1 w) c', h1=2, w1=2, b=11)
sio.savemat(PATH + name,{'output':result})
