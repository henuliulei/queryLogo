import torch
from torch import nn as nn
import torchvision as tv
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        # 中间扩展层的通道数
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            # 1x1的逐点卷积进行升维
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 深度可分离模块
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 1x1的逐点卷积进行降维
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.stride == 1:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 标准卷积
        def conv_bn(dim_in, dim_out, stride):
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, stride, 1, bias=False),
                nn.BatchNorm2d(dim_out),
                nn.ReLU6(inplace=True)
            )

        # 深度分解卷积
        def conv_dw(dim_in, dim_out, stride):
            return nn.Sequential(
                InvertedResidual(dim_in, dim_in, stride, expand_ratio=6),
                nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(dim_out),
                nn.ReLU6(inplace=True)
            )

        # MobileNet 每一个卷积组的结构参数
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 2),
            #             conv_dw(512,512,2),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        '''UNet-Encoder:'''
        self.e_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )  # [128,128,64]
        self.e_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )  # [64,64,128]
        self.e_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )  # [32,32,256]
        self.e_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )  # [16,16,512]
        self.e_conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )  # [8,8,512]

        '''
         #  conditional-vgg16
        self.c_vgg = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # [32,32,64]

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # [16,16,128]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # [8,8,256]

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # [4,4,512]

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # [2,2,512]

            # nn.MaxPool2d(2, 2),  # [1,1,512]
            nn.Conv2d(512, 512, 1, 2, 0),
        )
        '''



        # MobileNet_v2用来替代conditional branch 的 vgg16
        # self.conditional = MobileNet()

        # decoder: #3x3 convolution + 3x3 convolution 注意还要在后面加上采样层（后接激活函数）
        self.d_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.d_conv2 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.d_conv3 = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.d_conv4 = nn.Sequential(
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.d_conv5 = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )

        # 1x1卷积
        self.conv1_1_1 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv1_1_2 = nn.Conv2d(768, 256, 1, 1, 0)
        self.conv1_1_3 = nn.Conv2d(640, 128, 1, 1, 0)
        self.conv1_1_4 = nn.Conv2d(576, 64, 1, 1, 0)

        self.conv_C = nn.Sequential(nn.Conv2d(512, 512, 1, 2, 0)) # conditional branch 提取特征的最后一部，主要是shape size from 2x2 to 1x1

    def _tile(self, vector):
        '''
        Tile operation
        input: 维度为[batch_size,512,1,1]的三维张量，在本例中是通过c_vgg网络后的query logo的特征表示
        output:list,元素维度依次分别为[batch_size,512,128,128],[batch_size,512,64,64],[batch_size,512,32,32],[batch_size,512,16,16],[batch_size,512,8,8]
        '''
        batch_size = vector.shape[0]
        Ctile128 = vector.expand(batch_size, 512, 128, 128)
        Ctile64 = vector.expand(batch_size, 512, 64, 64)
        Ctile32 = vector.expand(batch_size, 512, 32, 32)
        Ctile16 = vector.expand(batch_size, 512, 16, 16)
        Ctile8 = vector.expand(batch_size, 512, 8, 8)

        # Ctile128 = vector.repeat(1, 1, 32, 32)
        # Ctile64 = vector.repeat(1, 1, 16, 16)
        # Ctile32 = vector.repeat(1, 1, 8, 8)
        # Ctile16 = vector.repeat(1, 1, 4, 4)
        # Ctile8 = vector.repeat(1,1,2,2)
        return [Ctile128, Ctile64, Ctile32, Ctile16, Ctile8]

    def forward(self, input1,input2):  # input1:query logo[batch_size,256,256,3];input2:target image[batch_size,64,64,3]

        # 下面这个条件分支（和segment分支的encoder部分共享权重）用来提取query image特征。与segmentation branch的encoder组成siamese network
        q_zi1 = self.e_conv1(input1)  # query logo的表征特征，zi是其在figure2中的符号  [batch_size,512,1,1]
        q_zi2 = self.e_conv2(q_zi1)
        q_zi3 = self.e_conv3(q_zi2)
        q_zi4 = self.e_conv4(q_zi3)
        q_zi5 = self.e_conv5(q_zi4)
        q_zi = self.conv_C(q_zi5)

        e_s1 = self.e_conv1(input2)  # target image在encoder第一阶段s=1的表征特征 [batch_size,64,128,128]
        e_s2 = self.e_conv2(e_s1)  # [batch_size,128,64,64]
        e_s3 = self.e_conv3(e_s2)  # [batch_size,256,32,32]
        e_s4 = self.e_conv4(e_s3)  # [batch_size,512,16,16]
        e_s5 = self.e_conv5(e_s4)  # [batch_size,512,8,8]

        # tile operation
        c_tile = self._tile(q_zi)
        c_tile128, c_tile64, c_tile32, c_tile16, c_tile8 = c_tile[0], c_tile[1], c_tile[2], c_tile[3], c_tile[4]

        # cat operation  [batch_size,512,128.128] cat [batch_size,64,128,128] =[batch_size,512+64,128,128]  沿着通道方向拼接
        e_cat128 = torch.cat((c_tile128, e_s1), dim=1)  # [batch_size,512+64,128,128]
        e_cat64 = torch.cat((c_tile64, e_s2), dim=1)  # [batch_size,512+128,64,64]
        e_cat32 = torch.cat((c_tile32, e_s3), dim=1)  # [batch_size,512+256,32,32]
        e_cat16 = torch.cat((c_tile16, e_s4), dim=1)  # [batch_size,512+512,16,16]

        # decoder 1*1 convolution
        d_fs1 = self.conv1_1_1(e_cat16)  # 命名d:decoder part  fs1:fs是论文中的表示符号; 1,即s=1，在decoder的第1阶段 与target image做cat的 [batch_size,512,16,16]
        d_fs2 = self.conv1_1_2(e_cat32)  # [batch_size,256,32,32]
        d_fs3 = self.conv1_1_3(e_cat64)  # [batch_size,128,64,64]
        d_fs4 = self.conv1_1_4(e_cat128)  # [batch_size,64,128,128]

        # decoder part
        d_fuse0 = torch.cat((e_s5, c_tile8), 1)

        d_s1 = F.relu(F.interpolate(self.d_conv1(d_fuse0), scale_factor=2, mode='bilinear',align_corners=True))  # [batch_size,512,16,16]

        d_fuse1 = torch.cat((d_s1, d_fs1), 1)  # [b,1024,16,16]
        d_s2 = F.relu(F.interpolate(self.d_conv2(d_fuse1), scale_factor=2, mode='bilinear', align_corners=True))  # [b,256,32,32]

        d_fuse2 = torch.cat((d_s2, d_fs2), 1)  # [b,512,32,32]
        d_s3 = F.relu(F.interpolate(self.d_conv3(d_fuse2), scale_factor=2, mode='bilinear', align_corners=True))  # [b,128,64,64]

        d_fuse3 = torch.cat((d_s3, d_fs3), 1)  # [b,256,64,64]
        d_s4 = F.relu(F.interpolate(self.d_conv4(d_fuse3), scale_factor=2, mode='bilinear', align_corners=True))  # [b,64,128,128]

        d_fuse4 = torch.cat((d_s4, d_fs4), 1)  # [b,128,128,128]
        d_s5 = F.interpolate(self.d_conv5(d_fuse4), scale_factor=2, mode='bilinear', align_corners=True) # [b,1,256,256]

        # output
        output = d_s5
        return output
