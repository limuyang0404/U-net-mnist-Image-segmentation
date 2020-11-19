# coding=UTF-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from os import listdir
import random
from os.path import isfile, join, getsize
'''this function is made to get a Two-dimensional dimensions with mirrored edges'''
# def label_img_into_label(label_img):
#     class_color_coding = [
#         [0, 0, 0],
#         [156, 0, 147],  # blue
#         [237, 29, 37],  # red
#         [0, 255, 0],  # green
#         [0, 255, 255],  # cyan
#         [255, 0, 255],  # blue
#         [255, 255, 0],
#         [255, 246, 143],
#         [255, 193, 193],
#         [25, 106, 106],
#         [99, 184, 255]  # yellow
#     ]
#     img = plt.imread(label_img)
#     img = np.uint8(img[:, :, :]*255)
#     img_row = img.shape[0]
#     img_col = img.shape[1]
#     label = np.zeros((1, img_row, img_col))
#     for i in range(img_row):
#         for j in range(img_col):
#             for k in range(11):
#                 if img[i, j, 0]==class_color_coding[k][0] and img[i, j, 1]==class_color_coding[k][1] and img[i, j, 2]==class_color_coding[k][2]:
#                 # if np.uint8(img[i, j, :]*255).any()==np.array(class_color_coding[k]).any():
#                     label[0, i, j] = k
#     label = Variable(torch.Tensor(label).long())
#     return label

def label_img_into_label(file_path, file_name_list):
    class_color_coding = [
        [0, 0, 0],
        [156, 0, 147],  # blue
        [237, 29, 37],  # red
        [0, 255, 0],  # green
        [0, 255, 255],  # cyan
        [255, 0, 255],  # blue
        [255, 255, 0],
        [255, 246, 143],
        [255, 193, 193],
        [25, 106, 106],
        [99, 184, 255]  # yellow
    ]
    img_example = plt.imread(join(file_path, file_name_list[0]))
    img_row = img_example.shape[0]
    img_col = img_example.shape[1]
    img_list = []
    label = np.zeros((len(file_name_list), img_row, img_col))
    for m in range(len(file_name_list)):
        img = plt.imread(join(file_path, file_name_list[m]))
        img = np.uint8(img[:, :, :]*255)
        img_row = img.shape[0]
        img_col = img.shape[1]
        for i in range(img_row):
            for j in range(img_col):
                for k in range(11):
                    if img[i, j, 0]==class_color_coding[k][0] and img[i, j, 1]==class_color_coding[k][1] and img[i, j, 2]==class_color_coding[k][2]:
                        # if np.uint8(img[i, j, :]*255).any()==np.array(class_color_coding[k]).any():
                        label[m, i, j] = k
    label = Variable(torch.Tensor(label).long())
    return label

def get_data(file_path, file_name_list):
    img_list = []
    for i in range(len(file_name_list)):
        img = plt.imread(join(file_path, file_name_list[i]))
        img_list.append(img[np.newaxis, np.newaxis, :])
    for j in range(1, len(file_name_list)):
        img_list[0] = np.concatenate((img_list[0], img_list[j]), axis=0)
    data = Variable(torch.Tensor(img_list[0]).float())
    return data

def get_single_data(file_path):
    img = plt.imread(file_path)
    img = img[np.newaxis, np.newaxis, :]
    output = Variable(torch.Tensor(img).float())
    return output

# def get_data(file_path):
#     img = plt.imread(file_path)
#     img_size = img.shape
#     data = img[np.newaxis, np.newaxis, :]
#     data = Variable(torch.Tensor(data).float())
#     return data





if __name__ == '__main__':
    b = get_data(r"imgs/mnist_train_jpg_6000_mirror", ['0_140_1495.png', '0_121_1310.png', '0_109_1093.png'])
    print(b, type(b), b.size())
    c = label_img_into_label(r"imgs/mnist_train_jpg_6000_classify_mirror", ['0_121_1310.png', '0_109_1093.png', '7_305_2754.png', '4_269_2439.png'])
    print(c, type(c), c.size())

