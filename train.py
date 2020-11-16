# coding=UTF-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
import random
from os.path import isfile, join, getsize
from label_img_into_label import label_img_into_label, get_data
from texture_net import TextureNet
import torch
from torch import nn
import torch.nn.functional as F

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
optimizer = torch.optim.Adam(network.parameters(), lr=0.03)#Adam method
# optimizer.load_state_dict(torch.load(join('F3','optimizer.pth')))
checkpoint = {'model':network.state_dict(),'optimizer':optimizer.state_dict()}
model_path = 'saved_model.pt'
optimizer_path = 'optimizer.pth'
# network.module.load_state_dict(torch.load(model_path))
img_path = "imgs/mnist_train_jpg_6000_mirror"
label_img_path = "imgs/mnist_train_jpg_6000_classify_mirror"


for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()
# optimizer.load_state_dict(torch.load(optimizer_path))
loss_list = []
for z in range(20):
    files = [f for f in listdir(img_path) if
             isfile(join(img_path, f))]
    time_list = 0
    file_counter = 0
    data_list = []
    for file in files:
        file_counter += 1
        data_list.append(file)
        if file_counter == 20:
            time_list += 1
            data = get_data(img_path, data_list)
            label = label_img_into_label(label_img_path, data_list)
            data = data.to(device)
            label = label.to(device)
            network.train()
            output = network(data)
            loss = cross_entropy(output, label)
            print(r"The %d epoch's %d loss is:" % (z, time_list), loss)
            loss_list.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            file_counter = 0
            data_list = []
    torch.save(network.module.state_dict(), 'saved_model.pt')  # 网络保存为saved_model.pt
    torch.save(optimizer.state_dict(), 'optimizer.pth')

x1 = range(len(loss_list))
plt.plot(x1, loss_list)
plt.savefig('loss.png')

