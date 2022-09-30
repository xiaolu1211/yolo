# 定义目标检测损失
class YOLOLoss(nn.Module):
    def __init__(self,anchors,num_classes,input_shape,cuda,anchors_mask=[[6,7,8],[3,4,5],[0,1,2]]):
        super(YOLOLoss,self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]，大感受野对应大框
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors=anchors
        self.num_classes=num_classes
        self.bbox_attrs=5+num_classes
        self.input_shape=input_shape
        self.anchors_mask=anchors_mask
        
        self.giou=True
        self.balance=[0.4,1.0,4]
        self.box_ratio=0.05
        self.obj_ratio=5*(input_shape[0]*input_shape[1])/(416**2)
        self.cls_ratio=1*(num_classes/80)
        
        self.ignore_threshold=0.5
        self.cuda=cuda
    def clip_by_tensor(self,t,t_min,t_max):
        t=t.float()
        result=(t>=t_min).float()*t+(t<t_min).float()*t_min
        result=(result<=t_max).float()*result+(result>t_max).float()*t_max
        return result
    
    def MSELoss(self,pred,target):
        return torch.pow(pred-target,2)
       #BCELOSS用于计算多标签分类损失 
    def BCELoss(self,pred,target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred,epsilon,1-epsilon)
        output=-target*torch.log(pred) -(1.0-target)*torch.log(1.0-pred)
        return output
        #计算两个框之间的giou
    def box_giou(self,b1,b2):
#             """
#         输入为：
#         ----------
#         b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
#         b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
#         返回为：
#         -------
#         giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
#         """
        #----------------------------------------------------#
        #   求出预测框左上角右下角
        #----------------------------------------------------#
        b1_xy=b1[...,:2]
        b1_wh=b1[...,2:4]
        b1_wh_half=b1_wh/2
        b1_mins=b1_xy-b1_wh/2
        b1_maxes=b1_xy+b1_wh_half
            
        b2_xy=b2[...,:2]
        b2_wh=b2[...,2:4]
        b2_wh_half=b2_wh/2
        b2_mins=b2_xy-b2_wh/2
        b2_maxes=b2_xy+b2_wh_half
            
        intersect_mins=torch.max(b1_mins,b2_mins)
        intersect_maxes=torch.min(b1_maxes,b2_maxes)
        intersect_wh=torch.max(intersect_maxes-intersect_mins,torch.zeros_like(intersect_maxes))
        intersect_areas=intersect_wh[...,0]*intersect_wh[...,1]
        b1_area=b1_wh[...,0]*b1_wh[...,1]
        b2_area=b2_wh[...,0]*b2_wh[...,1]
        union_area=b1_area+b2_area-intersect_areas
        iou=intersect_areas/union_area
             #----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        #----------------------------------------------------#
        enclose_mins=torch.min(b1_mins,b2_mins)
        enclose_maxes=torch.max(b1_maxes,b2_maxes)
        enclose_wh=torch.max(enclose_maxes-enclose_mins,torch.zeros_like(intersect_maxes))
            
        enclose_area=enclose_wh[...,0]*enclose_wh[...,1]                #包含两个框的最小框的面积
        giou=iou-(enclose_area-union_area)/enclose_area
            
        return giou
        
    def forward(self,l,inputs,targets=None):
        #----------------------------------------------------#
        #   l代表的是，当前输入进来的有效特征层，是第几个有效特征层
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets代表的是真实框。
        #----------------------------------------------------#
        #--------------------------------#
        #   获得图片数量，特征层的高和宽
        #   13和13
        #--------------------------------#
        bs=inputs.size(0)
        in_h=inputs.size(2)
    
        in_w=inputs.size(3)
             #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        #   stride_h和stride_w都是32。
        #-----------------------------------------------------------------------#
        stride_h=self.input_shape[0]/in_h          #
        stride_w=self.input_shape[1]/in_w
            #相对于特征层的锚框大小
        scaled_anchors=[(a_w/stride_w,a_h/stride_h) for a_w,a_h in self.anchors]
             #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   bs, 3*(5+num_classes), 13, 13 => batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #-----------------------------------------------#
        prediction=inputs.view(bs,len(anchors_mask[l]),self.bbox_attrs,in_h,in_w).permute(0,1,3,4,2).contiguous()
            
        x=torch.sigmoid(prediction[...,0])
        y=torch.sigmoid(prediction[...,1])
            
        w=prediction[...,2]
        h=prediction[...,3]
            
        conf=torch.sigmoid(prediction[...,4])
            
        pred_cls=torch.sigmoid(prediction[...,5:])
            
        y_true,noobj_mask,box_loss_scale = self.get_target(l,targets,scaled_anchors,in_h,in_w)
            
        noobj_mask,pred_boxes=self.get_ignore(l,x,y,h,w,targets,scaled_anchors,in_h,in_w,noobj_mask)
            
        if self.cuda:
            y_true=y_true.type_as(x)
            noobj_mask      = noobj_mask.type_as(x)
            box_loss_scale  = box_loss_scale.type_as(x)
        #--------------------------------------------------------------------------#
        #   box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
        #   2-宽高的乘积代表真实框越大，比重越小，小框的比重更大。
        #--------------------------------------------------------------------------#     
        box_loss_scale=2-box_loss_scale
            
        loss=0
        obj_mask=y_true[...,4]==1   #真实框置信度为1的框代表这个单元格内有样本需要检测，应为正样本
        n=torch.sum(obj_mask)
            
        if n!=0:
            if self.giou:
                giou=self.box_giou(pred_boxes,y_true[...,:4]).type_as(x)
                loss_loc=torch.mean((1-giou)[obj_mask])
            else:
                loss_x=torch.mean(self.BCELoss(x[obj_mask],y_true[...,0][obj_mask])*box_loss_scale[obj_mask])
                loss_y=torch.mean(self.BCELoss(y[obj_mask],y_true[...,1][obj_mask])*box_loss_scale[obj_mask])
                    
                loss_w=torch.mean(self.MSELoss(w[obj_mask],y_true[...,2][obj_mask])*box_loss_scale[obj_mask])
                loss_h=torch.mean(self.MSELoss(h[obj_mask],y_true[...,3][obj_mask])*box_loss_scale[obj_mask])
                    
                loss_loc=(loss_x+loss_y+loss_w+loss_h)*0.1 #位置损失只计算正样本
                    
            loss_cls=torch.mean(self.BCELoss(pred_cls[obj_mask],y_true[...,5:][obj_mask]))#分类损失只计算正样本
            loss+=loss_loc*self.box_ratio+loss_cls*self.cls_ratio
            
        loss_conf=torch.mean(self.BCELoss(conf,obj_mask.type_as(conf))[noobj_mask.bool()|obj_mask]) #置信度损失需要计算正样本和负样本，但不参与计算的框不需要参加
        loss+=loss_conf*self.balance[l]*self.obj_ratio
            
        return loss
    def caculate_iou(self,_box_a,_box_b):
        b1_x1,b1_x2=_box_a[:,0]-_box_a[:,2]/2,_box_a[:,0]+_box_a[:,2]/2
        b1_y1,b1_y2=_box_a[:,1]-_box_a[:,3]/2,_box_a[:,1]+_box_a[:,3]/2
            
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
            
        box_a=torch.zeros_like(_box_a)
        box_b=torch.zeros_like(_box_b)
        box_a[:,0],box_a[:,1],box_a[:,2],box_a[:,3]=b1_x1,b1_y1,b1_x2,b1_y2
        box_b[:,0],box_b[:,1],box_b[:,2],box_b[:,3]=b2_x1,b2_y1,b2_x2,b2_y2
            
        A=box_a.size(0)
        B=box_b.size(0)
            
        max_xy=torch.min(box_a[:,2:].unsqueeze(1).expand(A,B,2),box_b[:,2:].unsqueeze(0).expand(A,B,2))
        min_xy=torch.max(box_a[:,:2].unsqueeze(1).expand(A,B,2),box_b[:,:2].unsqueeze(0).expand(A,B,2))
        inter=torch.clamp((max_xy-min_xy),min=0)
        inter=inter[:,:,0]*inter[:,:,1]
        area_a=((box_a[:,2]-box_a[:,0])*(box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter)
        area_b=((box_b[:,2]-box_b[:,0])*(box_b[:,3]-box_b[:,1])).unsqueeze(0).expand_as(inter)
            
        union=area_a+area_b-inter
            
        return inter/union   #返回的是A框中的每个框与B框中每个的交并比
           
    def get_target(self,l,targets,anchors,in_h,in_w):#用于获取和预测框同等尺寸的真实框，y_true，
            #输入的targets是含有物体框的经过归一化的框的中心点的坐标以及宽高
        bs=len(targets)
            #用于选取哪些先验框不包含物体，不参与位置以及类别损失的计算，不包含物体的先验框的单元值为1
        noobj_mask=torch.ones(bs,len(self.anchors_mask[l]),in_h,in_w,requires_grad=False)
            #   让网络更加去关注小目标
        box_loss_scale=torch.zeros(bs,len(self.anchors_mask[l]),in_h,in_w,requires_grad=False)
            
        y_true=torch.zeros(bs,len(self.anchors_mask[l]),in_h,in_w,self.bbox_attrs,requires_grad=False)
            
        for b in range(bs):
            if len(targets[b])==0:
                continue
            batch_target=torch.zeros_like(targets[b])
                
            batch_target[:,[0,2]]=targets[b][:,[0,2]]*in_w
            batch_target[:,[1,3]]=targets[b][:,[1,3]]*in_h
            batch_target[:,4]=targets[b][:,4]              #框内物体的种类
                #将真实框转换为以（0,0）为中心点，宽高的形式，num_true_box, 4
            gt_box=torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0),2)),batch_target[:,2:4]),1))
                #将先验框转换为以（0,0）为中心点，宽高的形式，9, 4
            anchor_shape=torch.FloatTensor(torch.cat((torch.zeros((len(anchors),2)),torch.FloatTensor(anchors)),1))
            #-------------------------------------------------------#
            #   计算交并比
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            #   best_ns:
            #   [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            #-------------------------------------------------------#
            best_ns=torch.argmax(self.caculate_iou(gt_box,anchor_shape),dim=-1)
                
            for t ,best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:  #判断与真实框重合度最大的先验框是否属于当前特征层的先验框
                    continue
                    
                k=self.anchors_mask[l].index(best_n)#   判断这个先验框是当前特征点的哪一个先验框
                    #   获得真实框属于哪个网格点
                i=torch.floor(batch_target[t,0]).long()
                j=torch.floor(batch_target[t,1]).long()
                     #   取出真实框的种类
                c=batch_target[t,4].long()
                    #   noobj_mask代表无目标的特征点
#                 print(k,j,i)
                noobj_mask[b,k,j,i]=0
                    
                if not self.giou:
                    y_true[b,k,j,i,0]=batch_target[t,0]-i.float()                  #距离中心的偏差值
                    y_true[b,k,j,i,1]=batch_target[t,1]-j.float()
                    y_true[b,k,j,i,2]=math.log(batch_target[t,2]/anchors[best_n][0])
                    y_true[b,k,j,i,3]=math.log(batch_target[t,3]/anchors[best_n][1])
                    y_true[b,k,j,i,4]=1
                    y_true[b,k,j,i,5+c]=1
                        
                else:
                    y_true[b,k,j,i,0]=batch_target[t,0]                           #特征层的中心坐标值
                    y_true[b,k,j,i,1]=batch_target[t,1]
                    y_true[b,k,j,i,2]=batch_target[t,2]
                    y_true[b,k,j,i,3]=batch_target[t,3]
                    y_true[b,k,j,i,4]=1
                    y_true[b,k,j,i,5+c]=1
                #----------------------------------------#
                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                #----------------------------------------#    
                box_loss_scale[b,k,j,i]=batch_target[t,2]*batch_target[t,3]/in_h/in_w
            return y_true,noobj_mask,box_loss_scale
    def get_ignore(self,l,x,y,h,w,targets,scaled_anchors,in_h,in_w,noobj_mask):
                #得到哪些预测框是可以忽略不参与计算的，即该框与真实框有一定的交并比但并不负责预测这个物体，该物体仅由最大交并比的框预测
        bs=len(targets)
                     #   生成网格，先验框中心，网格左上角
        grid_x=torch.linspace(0,in_w-1,in_w).repeat(in_h,1).repeat(int(bs*len(self.anchors_mask[l])),1,1).view(x.shape).type_as(x)
        grid_y=torch.linspace(0,in_h-1,in_h).repeat(in_w,1).t().repeat(int(bs*len(self.anchors_mask[l])),1,1).view(y.shape).type_as(x)
                    # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w=torch.Tensor(scaled_anchors_l).index_select(1,torch.LongTensor([0])).type_as(x)
        anchor_h=torch.Tensor(scaled_anchors_l).index_select(1,torch.LongTensor([1])).type_as(x)
        anchor_w=anchor_w.repeat(bs,1).repeat(1,1,in_h*in_w).view(w.shape)
        anchor_h=anchor_h.repeat(bs,1).repeat(1,1,in_h*in_w).view(h.shape)
                    
        pred_boxes_x=torch.unsqueeze(x+grid_x,-1)
        pred_boxes_y=torch.unsqueeze(y+grid_y,-1)
        pred_boxes_w=torch.unsqueeze(torch.exp(w)*anchor_w,-1)
        pred_boxes_h=torch.unsqueeze(torch.exp(h)*anchor_h,-1)
        pred_boxes=torch.cat([pred_boxes_x,pred_boxes_y,pred_boxes_w,pred_boxes_h],dim=-1)
                    
        for b in range(bs):
                        #   将预测结果转换一个形式
                        #   pred_boxes_for_ignore      num_anchors, 4
            pred_boxes_for_ignore=pred_boxes[b].view(-1,4)
                        
            if len(targets[b])>0:
                batch_target=torch.zeros_like(targets[b])
                        
                batch_target[:,[0,2]]=targets[b][:,[0,2]]*in_w
                batch_target[:,[1,3]]=targets[b][:,[1,3]]*in_h
                            #   计算真实框，并把真实框转换成相对于特征层的大小
                            #   gt_box      num_true_box, 4
                batch_target=batch_target[:,:4].type_as(x)
                            
                ach_ious=self.caculate_iou(batch_target,pred_boxes_for_ignore)
                             #   每个先验框对应真实框的最大重合度
                             #   anch_ious_max   num_anchors=1*先验框个数*in_w*in_h，这样值为0的为交并比大的，其中obj_mask中为1的为与真实框对应的，其余的就为忽略计算的，而nobj_mask值为1的就为负样本
                ach_ious_max,_=torch.max(ach_ious,dim=0)
                ach_ious_max=ach_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][ach_ious_max>self.ignore_threshold]=0
        return noobj_mask,pred_boxes
