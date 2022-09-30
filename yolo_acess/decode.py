#解码输出框
import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np
class DecodeBox():
    def __init__(self,anchors,num_classes,input_shape,anchors_mask=[[6,7,8],[3,4,5],[0,1,2]]):
        super(DecodeBox,self).__init__()
        self.anchors=anchors
        self.num_classes=num_classes
        self.bbox_attrs=5+num_classes
        self.input_shape=input_shape
        
        self.anchors_mask=anchors_mask
    def decode_box(self,inputs):
        outputs=[]
        for i,inp in enumerate(input):#   输入的input一共有三个，他们的shape分别是 batch_size, 255, 13, 13batch_size, 255, 26, 26 batch_size, 255, 52, 52
            batch_size=inp.size(0)
            input_height=inp.size(2)
            input_width=inp.size(3)
            
            stride_h=self.input_shape[0]/input_height
            stride_w=self.input_shape[1]/input_width
            #获得相对于特征蹭的锚框大小
            scaled_anchors=[(anchor_width/stride_w,anchor_height/stride_h) for anchor_width,anchor_height in self.anchors[self.anchors_mask[i]]]
            
            prediction=inp.view(batch_size,len(self.anchors_mask[i]),self.bbox_attrs,input_height,input_width).permute(0,1,3,4,2).contiguous()
            
            x=torch.sigmoid(prediction[...,0])
            y=torch.sigmoid(prediction[...,1])
            w=prediction[...,2]
            h=prediction[...,3]
            conf=torch.sigmoid(prediction[...,4])
            pred_cls=torch.sigmoid(prediction[...,5:])
            
            FloatTensor=torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor=torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            
            grid_x=torch.linspace(0,input_width-1,input_width).repeat(input_height,1).repeat(
                batch_size*len(self.anchors_mask[i]),1,1).view(x.shape).type(FloatTensor)
            grid_x=torch.linspace(0,input_width-1,input_width).repeat(input_height,1).repeat(
                batch_size*len(self.anchors_mask[i]),1,1).view(x.shape).type(FloatTensor)
            anchor_w=FloatTensor(scaled_anchors).index_select(1,LongTensor([0]))
            anchor_h=FloatTensor(scaled_anchors).index_select(1,LongTensor([1]))
            anchor_w=anchor_w.repeat(batch_size,1).repeat(1,1,input_height*input_width).view(w.shape)
            anchor_h=anchor_w.repeat(batch_size,1).repeat(1,1,input_height*input_width).view(h.shape)
            
            pred_boxes=FloatTensor(predictione[...,:4].shape)
            #特征层的中心坐标
            pred_boxes[...,0]=x.data+grid_x
            pred_boxes[...,1]=y.data+grid_y
              #特征层的宽高
            pred_boxes[...,2]=torch.exp(w.data)*anchor_w
            pred_boxes[...,3]=torch.exp(h.data)*anchor_h
            _scale=torch.Tensor([input_width,input_height,input_width,input_height]).type(FloatTensor)
            #对特征层的中心坐标以及宽高 将输出结果归一化成小数的形式
            output=torch.cat((pred_boxese.view(batch_size,-1,4)/_scale,conf.view(batch_size,-1,1),pred_cls.view(batch_size,-1,self.num_classes)),-1)
            outputs.append(output.data)
            return outputs
        def yolo_correct_boxes(self,box_xy,box_wh,input_shape,image_shape,letterbox_image):#将坐标转换到最初图像的位置中
            box_yx=box_xy[...,::-1]
            box_hw=box_wh[...,::-1]
            input_shape=np.array(input_shape)
            image_shape=np.array(image_shape)
            
            if letterbox_image:
                #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
                new_shape=np.round(image_shape*np.min(input_shape/image_shape))
                offest=(input_shape-new_shape)/2./input_shape
                scale=input_shape/new_shape
                
                box_yx=(box_yx-offest)*scale
                box_hw*=scale
            box_mins=box_yx-(box_hw/2.)
            box_maxs=box_yx+(box_hw/2.)
            boxes=np.concatenate([box_mins[...,0:1],box_mins[...,1:2],box_maxs[...,0:1],box_maxs[...,1:2]],axis=-1)
            boxes*=np.concatenate([image_shape,image_shape],axis=-1)
            return boxes
        #非极大值抑制进行框选择
        def non_max_suppression(self,prediction,num_classes,input_shape,image_shape,letterbox_image,conf_thres=0.5,nms_thres=0.4):
            box_corner=prediction.new(prediction.shape)
            #   将预测结果的格式转换成左上角右下角的格式。#   prediction  [batch_size, num_anchors, 85]
            box_corner[:,:,0]=prediction[:,:,0]-prediction[:,:,2]/2
            box_corner[:,:,1]=prediction[:,:,1]-prediction[:,:,3]/2
            box_corner[:,:,2]=prediction[:,:,0]+prediction[:,:,2]/2
            box_corner[:,:,3]=prediction[:,:,1]+prediction[:,:,3]/2
            pred_cls[:,:,:4]=box_corner[:,:,:4]
            output=[None for _ in range(len(prediction))]
            for i,image_pred in enumerate(prediction):#对每一张图片的预测框
                 #   对种类预测部分取max。得到预测框对应的种类
                class_conf,class_pred=torch.max(image_pred[:,5:5+num_classes],1,keepdim=True)
                #挑选种类置信度大于一定阈值的框 #   利用置信度进行第一轮筛选
                conf_mask=(image_pred[:,4]*class_conf[:,0]>=conf_thres).squeeze()
                image_pred=image_pred[conf_mask]
                class_conf=class_conf[conf_mask]
                class_pred=class_pred[conf_mask]
                if not image_pred.size(0):
                    continue
                #   detections  [num_anchors, 7]
                #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
                detections=torch.cat((image_pred[:,:5],class_conf.float(),class_pred.float()),1)
                #   获得预测结果中包含的所有种类
                unique_labels=detections[:,-1].cpu().unique()
                if prediction.is_cuda:
                    unique_labels = unique_labels.cuda()
                    detections = detections.cuda()
                
                for c in unique_labels:
                    #   获得某一类得分筛选后全部的预测结果
                    detections_class=detections[detections[:,-1]==c]
                    #   使用官方自带的非极大抑制会速度更快一些！
                    keep=nms(detections_class[:,:4],
                            detections_class[:,4]*detections_class[:,5],
                            nms_thres)
                    max_detections=detections_class[keep]
                    output[i]=max_detections if output[i] is None else torch.cat((output[i],max_detections))
                if output[i] is not None:
                    output[i]=output[i].cpu().numpy()
                    box_xy,box_wh=(output[i][:,0:2]+output[i][:,2:4]/2,output[i][:,2:4]-output[i][:,0:2])
                    output[i][:4]=self.yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape,letterbox_image)
            return output
            
