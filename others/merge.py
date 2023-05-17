import scipy.io as sio
import numpy as np
import cv2
from einops import rearrange

file_path = '/home/ubuntu/106-48t/personal_data/yj/local/results/cave11-0512.mat'
PATH = '/home/ubuntu/106-48t/personal_data/yj/local/merge'
data_dic = sio.loadmat(file_path)
result = data_dic['output']
#print(result.shape)#(44, 256, 256, 31)

result = rearrange(result, '(b h1 w1) h w c  -> b (h1 h) (w1 w) c', h1=2, w1=2, b=11)
# result = rearrange(result, '(b w1 h1) h w c  -> b (h1 h) (w1 w) c', h1=2, w1=2, b=11)
# result = rearrange(result, '(h1 w1 b) h w c  -> b (h1 h) (w1 w) c', h1=2, w1=2, b=11)
sio.savemat(PATH + '/cave11-0512.mat',{'output':result})

# unite = sio.loadmat('/home/ubuntu/yj/PSRT-main/results/cave-best.mat')
# unite = unite['output']
# print(unite.shape)

