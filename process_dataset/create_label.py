#voc格式生成标签文件
import os
import random
import xml.etree.ElementTree as ET
import numpy as np
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
annotation_mode=0
classes_path='D:/jupyter/yolo/VOCdevkit/VOC2007/voc_classes.txt'
trainval_percent=0.9#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
train_percent=0.9#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1

vocdevkit_path="D:/jupyter/yolo/VOCdevkit"#指向数据集所在文件
vocdevkit_sets=[('2007','train'),('2007','val')]#指向voc数据集
classes,_=get_classes(classes_path)
photo_num=np.zeros(len(vocdevkit_sets))#统计训练集与验证集中的图片数量
nums=np.zeros(len(classes))       #统计每一类别中的数量
def convert_annotation(year,image_id,list_file):#每一幅图片对应一个xml文件，读取该xml文件中的标注信息
    in_file=open(os.path.join(vocdevkit_path,"voc%s/Annotations/%s.xml"%(year,image_id)),encoding='utf-8')#标注文件的路径
    tree=ET.parse(in_file)   #将xml文件解析成树                                                                             #解析xml文件
    root=tree.getroot()      #获取根节点
    for obj in root.iter('object'): #得到每一个框的信息，查找指定节点‘object’
        diffcult=0
        if obj.find('diffcult')!=None: #查找指定节点‘object’下的‘difficult’节点
            diffcult=obj.find('difficult').text
        cls=obj.find('name').text#查找指定节点‘object’下的‘name’节点获取类别信息
        if cls not in classes or int(diffcult)==1:
            continue
        cls_id=classes.index(cls)
        xmlbox=obj.find('bndbox')#查找指定节点‘object’下的‘bndbox’节点获取框信息
        b=(int(float(xmlbox.find('xmin').text)),int(float(xmlbox.find('ymin').text)),int(float(xmlbox.find('xmax').text)),int(float(xmlbox.find('ymax').text)))
        list_file.write(' '+','.join([str(a) for a in b])+','+str(cls_id))
        
        nums[classes.index(cls)]=nums[classes.index(cls)]+1

random.seed(0)
if ' ' in os.path.abspath(vocdevkit_path): #路径与图片名称不能包含空格，空格用于后面划分
    raise ValueError('数据集中存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改')
if annotation_mode==0 or annotation_mode==1:#生成训练验证集、训练集、验证集、测试集的图像名称
    print('generate txt in imagesets.')
    xmlfilepath=os.path.join(vocdevkit_path,"VOC2007/Annotations")      #标签文件地址
    savebasepath=os.path.join(vocdevkit_path,"VOC2007/ImageSets/Main")#保存路径
    temp_xml=os.listdir(xmlfilepath)
    total_xml=[]
    for xml in temp_xml:
        if xml.endswith('.xml'):
            total_xml.append(xml)
    num=len(total_xml)                     #图片总数，每一张图片对应一个xml文件
    l=range(num)
    tv=int(num*trainval_percent)         #训练集验证集图片数量
    tr=int(tv*train_percent)             #训练集图片数量
    trainval=random.sample(list,tv)
    train=random.sample(trainval,tr)
    print('train and val size',tv)
    print('train size',tr)
    
    ftrainval=open(os.path.join(savebasepath,'trainval.txt'),'w')
    ftest=open(os.path.join(savebasepath,'test.txt'),'w')
    ftrain=open(os.path.join(savebasepath,'train.txt'),'w')
    fval=open(os.path.join(savebasepath,'val.txt'),'w')
    
    for i in l:
        name=total_xml[i][:-4]+'\n'   #将图片名称记录到对应的文本文件中
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print('generate txt in imagesets done.')
if annotation_mode==0 or annotation_mode==2:#生成训练集与验证集读取文本文档
    print('generate 2007_train.txt and 2007_val.txt for train.')
    type_index=0
    for year,image_set in vocdevkit_sets:
        #读取对应数据集中的图片名称
        image_ids=open(os.path.join(vocdevkit_path,"VOC%s/ImageSets/Main/%s.txt"%(year,image_set)),encoding='utf-8').read().strip().split()
        list_file=open('%s_%s.txt'%(year,image_set),'w',encoding='utf-8')
        #将信息写入
        for image_id in image_ids:
#             list_file.write("%s/VOC%s/JPEGImages/%s.jpg"%(os.path.abspath(vocdevkit_path),year,image_id))#写入图片保存路径
            list_file.write("%s/VOC%s/JPEGImages/%s.jpg"%(vocdevkit_path,year,image_id))#写入图片保存路径
            convert_annotation(year,image_id,list_file)#写入图片对应的标签框信息，并以空格进行分割
            list_file.write('\n')
        photo_num[type_index]=len(image_ids) #不同数据集下的图片总数
        type_index+=1
        list_file.close()
    print('generate 2007_train.txt and 2007_val.txt for train done')
    #打印类别名称与标签的对应关系
    def printTable(list1,list2):
        for i in range(len(list1[0])):
            print('|',end=' ')
            for j in range(len(list1)):
                print(list1[j][i].rjust(int(list2[j])),end=' ')
                print('|',end=' ')
            print()
    
    str_nums=[str(int(x)) for x in nums]
    tableDate=[classes,str_nums]
    colwidth=[0]*len(tableDate)
    len1=0
    for i in range(len(tableDate)):
        for j in range(len(tableDate[i])):
            if len(tableDate[i][j])>colwidth[i]:
                colwidth[i]=len(tableDate[i][j])
    printTable(tableDate,colwidth)
    
    if photo_num[0]<=500:
        print('训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代以满足足够的梯度下降次数')
    if np.sum(nums)==0:
        print('在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！')
        
