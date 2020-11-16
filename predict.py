# coding=UTF-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
import random
from os.path import isfile, join, getsize
from label_img_into_label import label_img_into_label, get_data, get_single_data
from texture_net import TextureNet
import torch
from torch import nn
'''this function is made to get a Two-dimensional dimensions with mirrored edges'''
network = TextureNet(n_classes=11)
cross_entropy = nn.CrossEntropyLoss() #Softmax function is included
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())
#Transfer model to gpu
if torch.cuda.device_count()>1:
    network = nn.DataParallel(network)
network.to(device)
# network.load_state_dict = (checkpoint)    #pytorch调用先前的模型参数
network.eval()
optimizer = torch.optim.Adam(network.parameters(), lr=0.003)#Adam method
# optimizer.load_state_dict(torch.load(join('F3','optimizer.pth')))
checkpoint = {'model':network.state_dict(),'optimizer':optimizer.state_dict()}
model_path = 'saved_model.pt'
optimizer_path = 'optimizer.pth'
network.module.load_state_dict(torch.load(model_path))
# img_path = "imgs/mnist_train_jpg_6000_mirror"
img_path = "imgs/mnist_test_jpg_1000_mirror"
# img_path = "imgs"
label_img_path = "imgs/mnist_train_jpg_6000_classify_mirror"
def predict(net_output):
    net_output = nn.Softmax(dim=1)(net_output)
    net_output = torch.squeeze(net_output)
    net_output = net_output.cpu()
    array = (net_output).detach().numpy()
    tensor_size0, tensor_size1, tensor_size2 = net_output.size(0), net_output.size(1), net_output.size(2)
    predict = np.zeros((tensor_size1, tensor_size2))
    for i in range(tensor_size1):
        for j in range(tensor_size2):
            value = 0
            for k in range(tensor_size0):
                print(i, j, k, array[k, i, j])
                value += array[k, i, j]*k       #like a vote process
            predict[i, j] = value
    plt.imshow(predict)
    plt.show()
    return

for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()
optimizer.load_state_dict(torch.load(optimizer_path))

data = get_single_data(join(img_path, '5_30_317.png'))
# label = label_img_into_label(join(label_img_path, '9_436_4350.png'))
data = data.to(device)
# label = label.to(device)
network.eval()
output = network(data)
predict(output)