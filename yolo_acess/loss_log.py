#yolo,评估，记录损失
import datetime
import os
import torch
import matplotlib
matplotlib.use('Agg')#运行不会显示图
import scipy .signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

# from .utils import cvtColor, preprocess_input, resize_image
# from .utils_bbox import DecodeBox
# from .utils_map import get_coco_map, get_map

class LossHistory():
    def __init__(self,log_dir,model,input_shape):
        self.log_dir=log_dir
        self.losses=[]
        self.val_loss=[]
        os.makedirs(self.log_dir)
        self.writer=SummaryWriter(self.log_dir)#写日志文件
        try:
            dummy_input=torch.randn(2,3,input_shape[0],input_shape[1])
            self.writer.add_graph(model,dummy_input)#在日志文件中，添加模型和输入
        except:
            pass
    def append_loss(self,epoch,loss,val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir,'epoch_loss.txt'),'a') as f:
            f.write(str(loss))
            f.write('\n')
        with open(os.path.join(self.log_dir,'epoch_val_loss.txt'),'a') as f:
            f.write(str(val_loss))
            f.write('\n')
        self.writer.add_scaler('loss',loss,epoch)#在日志文件中添加损失数据
        self.writer.add_scaler('val_loss',val_loss,epoch)
        self.loss_plot()
    def loss_plot(self):#画损失函数
        iters=range(len(self.losses))
        
        plt.figure()
        plt.plot(iters,self.losses,'red',linewidth=2,label='train loss')
        plt.plot(iters,self.val_loss,'coral',linewidth=2,label='val loss')
        try: 
            if len(self.losses)<25:
                num=5
            else:
                num=15
            #采用滤波器对信号进行平滑处理，去除噪声scipy.signal.savgol_filter(信号，窗长，多项式拟合的阶数)
            plt.plot(iters,scipy.signal.savgol_filter(self.losses,num,3),'green',linestyle='--',linewidth=2,label='smooth train loss')
            plt.plot(iters,scipy.signal.savgol_filter(self.val_loss,num,3),'#8B4513',linestyle='--',linewidth=2,label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        
        plt.savefig(os.path.join(self.log_dir,'epoch_loss.png'))
        
        plt.cla()
        plt.close('all')
class EvalCallback():
    def __init__(self,net,input_shape,anchors,anchors_mask,class_names,num_classes,val_lines,log_dir,cuda,map_out_path='.temp_map_out',
                max_boxes=100,confidence=0.05,nms_iou=0.5,letterbox_image=True,MINOVERLAP=0.5,eval_flag=True,period=1):
        super(EvalCallback,self).__init__()
        self.net=net
        self.input_shape=input_shape
        self.anchors=anchors
        self.anchors_mask=anchors_mask
        self.class_names=class_names
        self.num_classes=num_classes
        self.val_lines=val_lines
        self.log_dir=log_dir
        self.cuda=cuda
        self.map_out_path=map_out_path
        self.max_boxes=max_boxes
        self.confidence=confidence
        self.nms_iou=nms_iou
        self.letterbox_image=letterbox_image
        self.MINOVERLAP=MINOVERLAP
        self.eval_flag=eval_flag
        self.period=period
        
        self.bbox_util=DecodeBox(self.anchors,self.num_classes,(self.input_shape[0],input_shape[1]),self.anchors_mask)
        
        self.maps=[0]
        self.epoches=[0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir,'epoch_map.txt'),'a')as f:
                f.write(str(0))
                f.write('\n')
    def get_map_txt(self,image_id,image,class_names,map_out_path):#写检测结果的映射文件
        f=open(os.path.join(map_out_path,'detection-results/'+image_id+'.txt'),'w',encoding='utf-8')
        image_shape=np.array(np.shape(image)[0:2])
        image=cvtColor(image)
        #   给图像增加灰条，实现不失真的resize
        image_data=resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)
        #   添加上batch_size维度
        image_data=np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),(2,0,1)),0)
        
        with torch.no_grad:
            images=torch.from_numpy(image_data)
            if self.cuda:
                images-=images.cuda()
            outputs=self.net(images)#   将图像输入网络当中进行预测
            outputs=self.bbox_util.decode_box(outputs)#对框进行解码
             #   将预测框进行堆叠，然后进行非极大抑制
            results=bbox_util.non_max_suppresion(torch.cat(outputs,1),self.num_classes,self.input_shape,image_shape,self.letterbox_image,
                                                conf_thres=self.confidence,nms_thres=self.nms_iou)
            if results[0] is None:
                return
            top_label=np.array(results[0][:,6],dtype='int32')
            top_conf=results[0][:,4]*results[0][:,5]
            top_boxes=results[0][:,:4]
        top_100=np.argsort(top_conf)[::-1][:self.max_boxes]#根据置信度的大小挑选出一定数量的预测框
        top_boxes=top_boxes[top_100]
        top_conf=top_conf[top_100]
        top_label=top_label[top_100]
        
        for i,c in list(enumerate(top_label)):
            predicted_class=self.class_names[int(c)]
            box=top_boxes[i]
            score=str(top_conf[i])
            
            top,left,bottom,right=box
            if predicted_class not in class_names:
                continue
            #在映射文件中写入类别，置信度，框的位置信息
            f.write('%s %s %s %s %s %s\n' %(predicted_class,score[:6],str(int(left)),str(int(top)),str(int(right)),str(int(bottom))))
        f.close()
        return
    
    def on_epoch_end(self,epoch,model_eval):
        if epoch%self.period==0 and self.eval_flag:
            self.net=model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path,'ground-truth')):
                os.makedirs(os.path.join(self.map_out_path,'ground-truth'))
            if not os.path.exists(os.path.join(self.map_out_path,'detection-result')):
                os.makedirs(os.path.join(self.map_out_path,'detection-result'))
            print('Get map')
          
            for annotation_line in tqdm(self.val_lines):
                line=annotation_line.split()
                image_id=os.path.basename(line[0]).split('.')[0]
                image=Image.open(line[0])
                gt_boxes=np.array([np.array(list(map(int,box.split(','))))for box in line[1:]])
                self.get_map_txt(image_id,image,self.class_names,self.map_out_path)
                
                with open(os.path.join(self.map_out_path,'ground-truth/'+image_id+'.txt'),'w')as new_f:  ##写真实框的映射文件
                    for box in gt_boxes:
                        left,top,right,bottom,obj=box
                        obj_name=self.class_names[obj]
                        new_f.write('%s %s %s %s %s\n'%(obj_name,left,top,right,bottom))
#             print('Calculate Map.')
#             try:
#                 temp_map=get_coco_map(class_names=self.class_names,path=self.map_out_path)[1]
#             except:
#                 temp_map=get_map(self.MINOVERLAP,False,path=self.map_out_path)
                
#             self.maps.append(temp_map)
#             self.epoches.append(epoch)
#             with open(os.path.join(self.log_dir,'epoch_map.txt'),'a')as f:
#                 f.write(str(temp_map))
#                 f.write('\n')
#             plt.figure()
#             plt.plot(self.epoches,self.maps,'red',linewidth=2,label='train map')
            
#             plt.grid(True)
#             plt.xlabel('Epoch')
#             plt.ylabel('Map %s'%str(self.MINOVERLAP))
#             plt.title('A Map Curve')
#             plt.legend(loc='upper right')
            
#             plt.savefig(os.path.join(self.log_dir,'epoch_map.png'))
#             plt.cla()
#             plt.close('all')
            
#             print('Get map done')
            shutil.rmtree(self.map_out_path)#递归删除文件夹下的所有子文件夹和子文件
            
