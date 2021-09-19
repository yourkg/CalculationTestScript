# -*- coding: utf-8 -*-
# modify by yourkg ，21/9/10
# modify by yourkg ，21/9/18 v0.2

import torch
import torch.nn as nn
import torchvision
import numpy as np
import time 
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

print("PyTorch版本: ",torch.__version__)
print("Torchvision版本: ",torchvision.__version__)
print("查找GPU: ",torch.cuda.is_available())




__all__ = ['ResNet50', 'ResNet101','ResNet152']


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        #卷积核为7 * 7,stride = 2 padding为3
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        #最大池化层 3*3,stride =2
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        #增加矩阵维度
        self.expansion = expansion
        #降采样
        self.downsampling = downsampling
        #构建
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)
        #残差
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)
        #层
        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)
        #均值池化
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #全连接
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        #1
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = ResNet152()
    #print(model)
    selectis=0#选择显卡
    if torch.cuda.is_available():
        print(">",torch.cuda.get_device_name())
        print(">",torch.cuda.device_count())
        selectis=int(input("输入要测试的显卡号："))    
        print(">",torch.cuda.get_device_properties(selectis))
        device = torch.device("cuda")
        torch.device('cuda', selectis)
    else:
        device = torch.device("cpu")
        print("程序没有读到显卡。。")
    a=int(input("输入整数的循环次数(1~n):"))
    b=input("输入整数计算量(1~n,建议输入128，256，512，越大显存占到越多)")
    inputx = torch.randn(int(b), 3, 224, 224).to(device)
    model=model.to(device)
    t0=time.time()
    while a>0:
        t1=time.time()
        out = model(inputx)
        t2=time.time()
        print(out.sum(),t2-t1)
        a-=1
    t3=time.time()
    print("运行时间",t3-t0)
    #inputx = torch.randn(1, 3, 224, 224)
    
