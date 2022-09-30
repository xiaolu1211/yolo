import math
from collections import OrderedDict
import torch.nn as nn
from functools import partial
import numpy as np
#构建特征提取网络
# 残差结构，利用1卷积下降通道数，3卷积提取特征并上升通道数，残差连接
class BasicBlock(nn.Module):
    def __init__(self,inplanes,planes):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(inplanes,planes[0],kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(planes[0])
        self.relu1=nn.LeakyReLU(0.1)
        self.conv2=nn.Conv2d(planes[0],planes[1],kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes[1])
        self.relu2=nn.LeakyReLU(0.1)
    
    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu1(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu2(out)
        out+=residual
        return out

class DarkNet(nn.Module):
    def __init__(self,layers):
        super(DarkNet,self).__init__()
        self.inplanes=32
        #416*416*3->416*416*32
        self.conv1=nn.Conv2d(3,self.inplanes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplanes)
        self.relu1=nn.LeakyReLU(0.1)
        
        #416*416*3->208*208*64
        self.layer1=self._make_layer_([32,64],layers[0])
        #208*208*64->104*104*128
        self.layer2=self._make_layer_([64,128],layers[1])
        #104*104*128->52*52*256
        self.layer3=self._make_layer_([128,256],layers[2])
        #52*52*256->26*26*512
        self.layer4=self._make_layer_([256,512],layers[3])
        #26*26*512->13*13*1024
        self.layer5=self._make_layer_([512,1024],layers[4])
        
        self.layers_out_filters=[64,128,256,512,1024]
        
        #权值初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
#         在每一个layer里面，首先用一个步长为2的3*3卷积进行下采样，然后进行残差结构的堆叠
        
    def _make_layer_(self,planes,blocks):
        layers=[]
        #下采样，步长为2，卷积核大小为3
        layers.append(('ds_conv',nn.Conv2d(self.inplanes,planes[1],kernel_size=3,stride=2,padding=1,bias=False)))
        layers.append(('ds_bn',nn.BatchNorm2d(planes[1])))
        layers.append(('ds_relu',nn.LeakyReLU(0.1)))
        #加入残差结构，残差结构的个数由blocks决定
        self.inplanes=planes[1]
        for i in range(blocks):
            layers.append(('residual_{}'.format(i),BasicBlock(self.inplanes,planes)))
        return nn.Sequential(OrderedDict(layers))
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        
        x=self.layer1(x)
        x=self.layer2(x)
        out3=self.layer3(x)    #输出特征层1->52*52*256
        out4=self.layer4(out3) #输出特征层2->26*26*512
        out5=self.layer5(out4) #输出特征层3->13*13*1024
        
        return out3,out4,out5
def darknet53():
    model=DarkNet([1,2,8,8,4])
    return model
#从特征层获取预测结果：1.构建FPN特征金字塔进行加强特征提取 2.利用YOLO Head对三个有效特征层进行预测
def conv2d(filter_in,filter_out,kernel_size):
    pad=(kernel_size-1)//2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ('conv',nn.Conv2d(filter_in,filter_out,kernel_size=kernel_size,stride=1,padding=pad,bias=False)),
        ('bn',nn.BatchNorm2d(filter_out)),
        ('relu',nn.LeakyReLU(0.1)),
        ]))
 # make_last_layers里面共有七个卷积，前五个用于提取特征，并上升构造金字塔，后两个用于获得yolo网络的预测结果
def make_last_layers(filters_list,in_filters,out_filters):
    m=nn.Sequential(
        conv2d(in_filters,filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        conv2d(filters_list[1],filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        conv2d(filters_list[1],filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        nn.Conv2d(filters_list[1],out_filters,kernel_size=1,stride=1,padding=0,bias=True)
    )
    return m
#yolo网络结构
class YoloBody(nn.Module):
    def __init__(self,anchors_mask,num_classes,pretrained=False):
        super(YoloBody,self).__init__()
        #生成特征提取网络darknet53，获取3个特征层
        self.backbone=darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load())
            
        #   out_filters : [64, 128, 256, 512, 1024]
        out_filters=self.backbone.layers_out_filters
        
        #计算yolohead的输出通道数，先验框个数*（类别+5）
        self.last_layer0=make_last_layers([512,1024],out_filters[-1],len(anchors_mask[0])*(num_classes+5))
        
        self.last_layer1_conv=conv2d(512,256,1)#上升特征降低通道数
        self.last_layer1_upsample=nn.Upsample(scale_factor=2,mode='nearest')#上采样，使经过特征提取的低层特征与上层特征尺寸一致
        
        self.last_layer1=make_last_layers([256,512],out_filters[-2]+256,len(anchors_mask[1])*(num_classes+5))
        
        self.last_layer2_conv=conv2d(256,128,1)
        self.last_layer2_upsample=nn.Upsample(scale_factor=2,mode='nearest')
        
        self.last_layer2=make_last_layers([128,256],out_filters[-3]+128,len(anchors_mask[2])*(num_classes+5))
        
    def forward(self,x):
            #获取三个特征层，他们的shape分别为 52*52*256，26*26*512，13*13*1024，
        x2,x1,x0=self.backbone(x)
            
            # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch=self.last_layer0[:5](x0)     #用于金字塔特征上升
            # 13,13,512 -> 13,13,1024 -> 13,13,先验框个数*（类别+5）
        out0=self.last_layer0[5:](out0_branch)   #第一个输出特征层 （bs*先验框个数*（类别+5）*13*13）
            
            # 13,13,512 -> 13,13,256 
        x1_in=self.last_layer1_conv(out0_branch)  #低层特征降低通道数
            #13,13,256 -> 26,26,256
        x1_in=self.last_layer1_upsample(x1_in)    #上采样，使经过特征提取的低层特征与上层特征尺寸一致
            
            # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in=torch.cat([x1_in,x1],1)             #特征拼接，在通道方向
            
            # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch=self.last_layer1[:5](x1_in)
            #  26,26,256 -> 26,26,512 -> 26,26,先验框个数*（类别+5）
        out1=self.last_layer1[5:](out1_branch)    #第二个输出特征层 （bs*先验框个数*（类别+5）*26*26）
            
             # 26,26,256 -> 26,26,128 
        x2_in=self.last_layer2_conv(out1_branch)
            #26,26,128 -> 52,52,128
        x2_in=self.last_layer2_upsample(x2_in)
            
            # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in=torch.cat([x2_in,x2],1)
            # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128-> 52,52,256 -> 先验框个数*（类别+5）*52*52
        out2=self.last_layer2(x2_in)              #第三个输出特征层 （bs*先验框个数*（类别+5）*52*52）
            
        return out0,out1,out2
            

        
