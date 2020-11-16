import torch
from torch import nn
import torch.nn.functional as F

class Conv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2d_block, self).__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.BatchNorm2d(out_channels))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.BatchNorm2d(out_channels))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)
    pass
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out
    pass
class TextureNet(nn.Module):
    def __init__(self,n_classes=2):
        super(TextureNet,self).__init__()
        self.left_conv_1 = Conv2d_block(in_channels=1, out_channels=32)
        self.pool_1 = nn.MaxPool2d(2, 2)                       #（16， 16， 16）
        self.drop1 = nn.Dropout(p=0.1)
        self.left_conv_2 = Conv2d_block(in_channels=32, out_channels=64)
        self.pool_2 = nn.MaxPool2d(2, 2)                        #（8， 8， 8）
        self.drop2 = nn.Dropout(p=0.1)
        self.left_conv_3 = Conv2d_block(in_channels=64, out_channels=128)
        self.pool_3 = nn.MaxPool2d(2, 2)                        #（4， 4， 4）
        self.drop3 = nn.Dropout(p=0.1)
        self.left_conv_4 = Conv2d_block(in_channels=128, out_channels=256)
        self.deconv_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=3, padding=3)
        self.right_conv_1 = Conv2d_block(in_channels=256, out_channels=128)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=3, padding=7)
        self.right_conv_2 = Conv2d_block(in_channels=128, out_channels=64)
        self.deconv_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.right_conv_3 = Conv2d_block(in_channels=64, out_channels=32)
        self.right_conv_4 = Conv2d_block(in_channels=32, out_channels=11)
    #Is called to compute network output
    def forward(self,x):
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)
        drop_1 = self.drop1(feature_1_pool)
        feature_2 = self.left_conv_2(drop_1)
        feature_2_pool = self.pool_2(feature_2)
        drop_2 = self.drop2(feature_2_pool)
        feature_3 = self.left_conv_3(drop_2)
        feature_3_pool = self.pool_3(feature_3)
        drop_3 = self.drop3(feature_3_pool)
        feature_4 = self.left_conv_4(drop_3)
        de_feature_1 = self.deconv_1(feature_4)
        temp1 = torch.cat((feature_3, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp1)
        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp2 = torch.cat((feature_2, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp2)
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        temp3 = torch.cat((feature_1, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp3)
        output = self.right_conv_4(de_feature_3_conv)
        return output
    def classify(self,x):
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)
        drop_1 = self.drop1(feature_1_pool)
        feature_2 = self.left_conv_2(drop_1)
        feature_2_pool = self.pool_2(feature_2)
        drop_2 = self.drop2(feature_2_pool)
        feature_3 = self.left_conv_3(drop_2)
        feature_3_pool = self.pool_3(feature_3)
        drop_3 = self.drop3(feature_3_pool)
        feature_4 = self.left_conv_4(drop_3)
        de_feature_1 = self.deconv_1(feature_4)
        temp1 = torch.cat((feature_3, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp1)
        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp2 = torch.cat((feature_2, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp2)
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        temp3 = torch.cat((feature_1, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp3)
        output = self.right_conv_4(de_feature_3_conv)
        return output
if __name__ == "__main__":
    x = torch.rand(size=(32, 1, 56, 56))
    net = TextureNet(n_classes=2)
    net.train()
    output = net.classify(x)
    print(type(output))
    print(output.size())
    print(type(output.size(0)))
    # out = test1.forward
    # out = torch.Tensor(out).float()
    # print(type(out))
    # print(out.size())
    # t = nn.Module()
    print(dir(nn.Module))