import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from predict import predict_area
from mnist_img_samples_mirror import mnist_img_mirror_single
from torch.autograd import Variable
import torch
def img_slice(img, tiny_size0, tiny_size1):
    img = plt.imread(img)
    size_0 = img.shape[0]
    size_1 = img.shape[1]
    # size_2 = img.shape[2]
    new_size_0 = math.ceil(size_0/tiny_size0)
    new_size_1 = math.ceil(size_1/tiny_size1)
    if size_0 % tiny_size0 != 0 or size_1 %tiny_size1 != 0:
        print("em... sorry,The img don't seem like a expected size.It will be filling by black area.")
    filling_img = np.zeros((new_size_0*tiny_size0, new_size_1*tiny_size1))
    filling_img[0:size_0, 0:size_1] = img
    tiny_img_list = []
    output_img_list = []
    for i in range(new_size_0):
        for j in range(new_size_1):
            tiny_img_list.append(filling_img[i*tiny_size0:i*tiny_size0+tiny_size0, j*tiny_size1:j*tiny_size1+tiny_size1])
    # for i in range(1, new_size_0*new_size_1):
    #     tiny_img_list[0] = np.concatenate((tiny_img_list[0], tiny_img_list[i]), axis=1)
    for i in range(new_size_0*new_size_1):
        img = tiny_img_list[i]
        img = mnist_img_mirror_single(img)
        img = img[np.newaxis, np.newaxis, :]
        output = Variable(torch.Tensor(img).float())
        print(output.size())
        predicted_img = predict_area(output)
        predicted_img = predicted_img[predicted_img.shape[0]//4:predicted_img.shape[0]//4*3, predicted_img.shape[1]//4:predicted_img.shape[1]//4*3]
        output_img_list.append(predicted_img)
    counter = 0
    output_img = np.zeros((new_size_0*tiny_size0, new_size_1*tiny_size1))
    for i in range(new_size_0):
        for j in range(new_size_1):
            output_img[i * tiny_size0:i * tiny_size0 + tiny_size0, j * tiny_size1:j * tiny_size1 + tiny_size1] = output_img_list[counter]
            counter += 1
    plt.imshow(output_img)
    plt.show()
    return
if __name__ == '__main__':
    img_slice(r"imgs/7914.png", 28, 28)