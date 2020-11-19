# coding=UTF-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
def mirroredge_2d(input_array, expand=2):
    if input_array.ndim == 2:
        im_size0 = input_array.shape[0]
        im_size1 = input_array.shape[1]
        array_expand = np.zeros([im_size0 * 3, im_size1 * 3])
        img_corner = np.flip(np.flip(input_array, 0), 1)
        img_edge0 = np.flip(input_array, 0)
        img_edge1 = np.flip(input_array, 1)
        array_expand[0:im_size0, 0:im_size1] = img_corner
        array_expand[im_size0:im_size0 * 2, 0:im_size1] = img_edge1
        array_expand[im_size0 * 2:im_size0 * 3, 0:im_size1] = img_corner
        array_expand[0:im_size0, im_size1:im_size1 * 2] = img_edge0
        array_expand[im_size0:im_size0 * 2, im_size1:im_size1 * 2] = input_array
        array_expand[im_size0 * 2:im_size0 * 3, im_size1:im_size1 * 2] = img_edge0
        array_expand[0:im_size0, im_size1 * 2:im_size1 * 3] = img_corner
        array_expand[im_size0:im_size0 * 2, im_size1 * 2:im_size1 * 3] = img_edge1
        array_expand[im_size0 * 2:im_size0 * 3, im_size1 * 2:im_size1 * 3] = img_corner
        array_expand = array_expand[im_size0 - expand:im_size0 * 2 + expand, im_size1 - expand:im_size1 * 2 + expand]
        return array_expand
    elif input_array.ndim == 3:
        im_size0 = input_array.shape[0]
        im_size1 = input_array.shape[1]
        im_size2 = input_array.shape[2]
        array_expand = np.zeros([im_size0*3, im_size1*3, im_size2])
        img_corner = np.flip(np.flip(input_array, 0), 1)
        img_edge0 = np.flip(input_array, 0)
        img_edge1 = np.flip(input_array, 1)
        array_expand[0:im_size0, 0:im_size1, :] = img_corner
        array_expand[im_size0:im_size0*2, 0:im_size1, :] = img_edge1
        array_expand[im_size0*2:im_size0*3, 0:im_size1, :] = img_corner
        array_expand[0:im_size0, im_size1:im_size1*2, :] = img_edge0
        array_expand[im_size0:im_size0*2, im_size1:im_size1*2, :] = input_array
        array_expand[im_size0*2:im_size0*3, im_size1:im_size1*2, :] = img_edge0
        array_expand[0:im_size0, im_size1*2:im_size1*3, :] = img_corner
        array_expand[im_size0:im_size0*2, im_size1*2:im_size1*3, :] = img_edge1
        array_expand[im_size0*2:im_size0*3, im_size1*2:im_size1*3, :] = img_corner
        array_expand = array_expand[im_size0-expand:im_size0*2+expand, im_size1-expand:im_size1*2+expand, :]
        return array_expand

def mnist_img_mirror(filename_in, filename_out):
    files = [f for f in listdir(filename_in) if
                isfile(join(filename_in, f))]
    m = 0
    for file in files:
        img = plt.imread(join(filename_in, file))
        if np.ndim(img)==2:
            img = mirroredge_2d(img, expand=img.shape[0]//2)
            im = Image.fromarray(img)
            im = im.convert('L')
            im.save(join(filename_out, file[0:-3] + 'png'))
            m += 1
            print('the', m, 'th img have been classified')
        elif np.ndim(img)==3:
            img = mirroredge_2d(img, expand=img.shape[0] // 2)
            im = Image.fromarray(np.uint8(img*255))
            im = im.convert('RGB')
            im.save(join(filename_out, file[0:-3] + 'png'))
            m += 1
            print('the', m, 'th img have been classified')
    return

def mnist_img_mirror_single(input_img):
    img = input_img
    img = mirroredge_2d(img, expand=img.shape[0] // 2)
    return img

if __name__ == '__main__':
    # mnist_img_mirror(r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\small_img_dir\mnist_train_jpg_6000', r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\small_img_dir\mnist_train_jpg_6000_mirror')
    # mnist_img_mirror(r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\small_img_dir\mnist_train_jpg_6000_classify', r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\small_img_dir\mnist_train_jpg_6000_classify_mirror')
    # mnist_img_mirror(r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\small_img_dir\mnist_test_jpg_1000', r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\small_img_dir\mnist_test_jpg_1000_mirror')
    # mnist_img_mirror(r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\small_img_dir\mnist_test_jpg_1000_classify', r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\small_img_dir\mnist_test_jpg_1000_classify_mirror')
    mnist_img_mirror(r"C:\Users\Administrator\Desktop\DIY-function\11", r"C:\Users\Administrator\Desktop\DIY-function\12")





