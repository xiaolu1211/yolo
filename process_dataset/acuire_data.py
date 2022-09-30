#从文本文件中读取数据
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
# from utils.utils import cvtColor,preprocess_input

class YoloDataset(Dataset):    
    def __init__(self,annotation_lines,input_shape,num_classes,train):
        super(YoloDataset,self).__init__()
        self.annotation_lines=annotation_lines  #从文本文件中读取的每行信息
        self.input_shape=input_shape
        self.num_classes=num_classes
        self.length=len(self.annotation_lines)#
        self.train=train
        
    def __len__(self):
        return self.length
    def __getitem__(self,index):
        index=index%self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image,box=self.get_random_data(self.annotation_lines[index],self.input_shape[0:2],random=self.train)#读取的每行的图片及对应标签
        image=np.transpose(preprocess_input(np.array(image,dtype=np.float32)),(2,0,1))#通道反转
        box=np.array(box,dtype=np.float32)
        if len(box)>0:#输入的targets是物体框的左下角和右上角，将其转换为中心点坐标以及狂傲的形式，并归一化
            box[:,[0,2]]=box[:,[0,2]]/self.input_shape[1]  #归一化
            box[:,[1,3]]=box[:,[1,3]]/self.input_shape[0]
            
            box[:,2:4]=box[:,2:4]-box[:,0:2]              #计算宽高
            box[:,0:2]=box[:,0:2]+box[:,2:4]/2
        return image,box
    def rand(self,a=0,b=1):
        return np.random.rand()*(b-a)+a
    #读取
    def get_random_data(self,annotation_lines,input_shape,jitter=.3,hue=.1,sat=0.7,val=0.4,random=True):
        line=annotation_lines.split()
        image=Image.open(line[0])#读入图像并转换成RGB图像
        image=cvtColor(image)
        iw,ih=image.size#图像宽高
        h,w=input_shape#目标宽高
        
        box=np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])#读取真实框信息，每个框内信息用，分隔；不同框用空格分隔
        
        if not random:#不随机增强
            scale=min(w/iw,h/ih)
            nw=int(iw*scale)
            nh=int(ih*scale)
            dx=(w-nw)//2
            dy=(h-nh)//2
            #   将图像多余的部分加上灰条
            image=image.resize((nw,nh),Image.BICUBIC)
            new_image=Image.new('RGB',(w,h),(128,128,128))
            new_image.paste(image,(dx,dy))
            image_data=np.array(new_image,np.float32)
             #   对真实框进行调整，将其转换在目标框下
            if len(box)>0:
                np.random.shuffle(box)
                box[:,[0,2]]=box[:,[0,2]]*nw/iw+dx
                box[:,[1,3]]=box[:,[0,2]]*nh/ih+dy
                box[:,0:2][box[:,0:2]<0]=0
                box[:,2][box[:,2]>w]=w
                box[:,3][box[:,3]>h]=h
                box_w=box[:,2]-box[:,0]
                box_h=box[:,3]-box[:,1]
                box=box[np.logical_and(box_w>1,box_h>1)]
            return image_data,box
        #随机增强
        #   对图像进行缩放并且进行长和宽的扭曲
        new_ar=iw/ih*self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale=self.rand(.25,2)
        if new_ar<1:
            nh=int(scale*h)
            nw=int(nh*new_ar)
        else:
            nw=int(scale*w)
            nh=int(nw*new_ar)
        image.resize((nw,nh),Image.BICUBIC)
        
        dx=int(self.rand(0,w-nw))
        dy=int(self.rand(0,h-nh))
        new_image=Image.new('RGB',(w,h),(128,128,128))
        new_image.paste(image,(dx,dy))
        image=new_image
        
        flip=self.rand()<.5
        if flip:image=image.transpose(Image.FLIP_LEFT_RIGHT)#  随机翻转图像
        
        image_data=np.array(image,np.uint8)
        #   对图像进行色域变换
        #   计算色域变换的参数
        r=np.random.uniform(-1,1,3)*[hue,sat,val]+1
        hue,sat,val=cv2.split(cv2.cvtColor(image_data,cv2.COLOR_RGB2HSV))#   将图像转到HSV上
        dtype=image_data.dtype
         #   应用变换
        x=np.arange(0,256,dtype=r.dtype)
        lut_hue=((x*r[0])%180).astype(dtype)
        lut_sat=np.clip(x*r[1],0,255).astype(dtype)
        lut_val=np.clip(x*r[2],0,255).astype(dtype)
        
        image_data=cv2.merge((cv2.LUT(hue,lut_hue),cv2.LUT(sat,lut_sat),cv2.LUT(val,lut_val)))
        image_data=cv2.cvtColor(image_data,cv2.COLOR_HSV2RGB)
        
        if len(box)>0:
                np.random.shuffle(box)
                box[:,[0,2]]=box[:,[0,2]]*nw/iw+dx
                box[:,[1,3]]=box[:,[0,2]]*nh/ih+dy
                box[:,0:2][box[:,0:2]<0]=0
                box[:,2][box[:,2]>w]=w
                box[:,3][box[:,3]>h]=h
                box_w=box[:,2]-box[:,0]
                box_h=box[:,3]-box[:,1]
                box=box[np.logical_and(box_w>1,box_h>1)]
        return image_data,box
#将每个批次的图像数据以及标签数据转换成Tensor格式    
def yolo_dataset_collate(batch):
    images=[]
    bboxes=[]
    for img,box in batch:
        images.append(img)
        bboxes.append(box)
    images=torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes=[torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images,bboxes                
