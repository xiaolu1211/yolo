#训练策略
#权重初始化
def weights_init(net,init_type='normal',init_gain=0.02):
    def init_func(m):
        classname=m.__class__.__name__
        if hasattr(m,'weight') and classname.find('Conv')!=-1:
            if init_type=='normal':
                torch.nn.init.normal_(m.weight.data,0.0,init_gain)
            elif init_type=='xavier':
                torch.nn.init.xavier_normal_(m.weight.data,gain=init_gain)
            elif init_type=='kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in')
            elif init_type=='orthogonal':
                torch.nn.init.normal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') !=-1:
            torch.nn.init.normal_(m.weight.data,1,0.02)
            torch.nn.init.constant_(m.weight.data,1,0.02)
        print('initialize network with %s type' %init_type)
        net.apply(init_func)
#学习率衰减策略
def get_lr_scheduler(lr_decay_type,lr,min_lr,total_iters,warmup_iter_ratio=0.05,warmup_lr_ratio = 0.1,no_aug_iter_ratio=0.05,step_num=10):
    def yolox_warm_cos_lr(lr,min_lr,total_iters,warmup_total_iters,warmup_lr_start,no_aug_iter,iters):
        if iters<=warmup_total_iters:
            lr=(lr-warmup_lr_start)*pow(iters/float(warmup_total_iters),2)+warmup_lr_start
        elif iters>total_iters-no_aug_iter:
            lr=min_lr
        else:
            lr=min_lr+0.5*(lr-min_lr)*(1.0+math.cos(math.pi*(iters-warmup_iter_iters)/(total_iters-warmup_total_iters-no_aug_iter)))
        return lr
    def step_lr(lr,decay_rate,step_size,iters):
        if step_size<1:
            raise ValueError('step_size must above 1.')
        n=iters//step_size
        out_lr=lr*decay_rate**n
        return out_lr
    if lr_decay_type=='cos':
        warmup_total_iters=min(max(warmup_iter_ratio*total_iters,1),3)
        warmup_lr_start=max(warmup_lr_ratio*lr,1e-6)
        no_aug_iter=min(max(no_aug_iter_ratio*total_iters,1),15)
        func=partial(yolox_warm_cos_lr,lr,min_lr,total_iters,warmup_total_iters,warmup_lr_start,no_aug_iter)
    else:
        decay_rate=(min_lr/lr)**(1/(step_num-1))
        step_size=total_iters/step_num
        func=partial(step_lr,lr,decay_rate,step_size)
    return func
def set_optimizer_lr(optimizer,lr_scheduler_func,epoch):
    lr=lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
#一个epoch里的训练过程
import os 
import torch
from tqdm import tqdm
# from utils.utils import get_lr

def fit_one_epoch(model_train,model,yolo_loss,loss_history,eval_callback,optimizer,epoch,epoch_step,epoch_step_val,gen,gen_val,Epoch,cuda,fp16,scaler,save_period,save_dir,local_rank=0):
    loss=0#训练损失
    val_loss=0#验证损失
    
    if local_rank==0:
        print('start train')
        pbar=tqdm(total=epoch_step,desc=f'Epoch{epoch+1}/{Epoch}',postfix=dict,mininterval=0.3)#进度条
    #训练过程
    model_train.train()
    for iteration,batch in enumerate(gen):
        if iteration>=epoch_step:
            break
        images,targets=batch[0],batch[1]
        with torch.no_grad():
            if cuda:
                images=images.cuda(local_rank)
                targets=[ann.cuda(local_rank) for ann in targets]
        optimizer.zero_grad()
        if not fp16:
            outputs=model_train(images)
            loss_value_all=0
            for l in range(len(outputs)):       #分别计算三个特征层的损失，相加得总损失
                loss_item=yolo_loss(l,outputs[l],targets)
                loss_value_all+=loss_item
            loss_value=loss_value_all
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs=model_train(images)
                loss_value_all=0
                for l in range(len(outputs)):
                    loss_item=yolo_loss(l,outputs[l],targets)
                    loss_value_all+=loss_item
                loss_value=loss_value_all
            
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        loss+=loss_value.item()
        if local_rank==0:
            pbar.set_postfix(**{'loss':loss/(iteration+1),
                               'lr':get_lr(optimizer)})
            pbar.update(1)
        
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    #模型验证
    model_train.eval()
    for iteration,batch in enumerate(gen_val):
        if iteration>=epoch_step_val:
            break
        images,targets=batch[0],batch[1]
        with torch.no_grad():
            if cuda:
                images=images.cuda(local_rank)
                targets=[ann.cuda(local_rank) for ann in targets]
            optimizer.zero_grad()
            outputs=model_train(images)
            loss_value_all=0
            for l in range(len(outputs)):
                loss_item=yolo_loss(l,outputs[l],targets)
                loss_value_all+=loss_item
            loss_value=loss_value_all
            val_loss+=loss_value.item()
            if local_rank==0:
                pbar.set_postfix(**{'val_loss':val_loss/(iteration+1)})
                pbar.update(1)
        if local_rank==0:
            pbar.close()
            print('Finish validation')
            loss_history.append_loss(epoch+1,loss/epoch_step,val_loss/epoch_step_val)
            eval_callback.on_epoch_end(epoch+1,model_train)
            print('Epoch:'+str(epoch+1)+'/'+str(Epoch))
            print('Total Loss:%.3f || val_loss: %.3f' %(loss/epoch_step,val_loss/epoch_step_val))
            
            #保存权值
            if (epoch+1)%save_period==0 or epoch+1==Epoch:
                torch.save(model.state_dict(),os.path.join(save_dir,'ep%03d-loss%.3f-val_loss%.3f.pth'%(epoch+1,loss/epoch_step,val_loss/epoch_step_val)))
            if len(loss_history.val_loss)<=1 or (val_loss/epoch_step_val)<=min(loss_history.val_loss):
                print('Save best model to best_epoch_weights.pth')
                torch.save(model.state_dict(),os.path.join(save_dir,'best_epoch_weights.pth'))
            
            torch.save(model.state_dict(),os.path.join(save_dir,'last_epoch_weights.pth'))
            
                                 
                                 
        

