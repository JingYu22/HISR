import scipy.io
import numpy as np
import cv2

file_path = '/home/ubuntu/yj/PSRT-main/merge/cave11-fuse.mat'
data_dic = scipy.io.loadmat(file_path)
# print(data_dic.keys())
dataset = data_dic['output']
dataset = dataset[2, :, :, :]
# print(dataset.shape)
array_rgb = dataset[:, :, [30,23,1]]*255
output = '/home/ubuntu/yj/PSRT-main/2.jpg'
cv2.imwrite(output, array_rgb)
