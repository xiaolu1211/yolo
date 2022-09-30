#处理数据集，根据json文件生成txt文件用于训练
import json
import os
from collections import defaultdict
train_dataset_path=''
val_dataset_path=''
train_annotation_path=''
train_annotation_path=''

train_output_path='train.txt'
val_output_path='val.txt'
#训练集
name_box_id=defaultdict(list)
id_name=dict()
f=open(train_annotation_path,encoding='utf-8')
data=json.load(f)

annotations=data['annotatuons']

for ant in annotations:
    id=ant['image_id']
    name=os.path.join(train_dataset_path,'%012d.jpg'%id)
    cat=ant['category_id']
    if cat >= 1 and cat <= 11:
         cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    name_box_id[name].append([ant['bbox'],cat])
f=open(train_output_path,'w')
for key in name_box_id.keys():#读取不同图片的标签信息
    f.write(key)
    box_infos=name_box_id[key]
    for info in box_infos:#读取一张图中不同框的具体信息，以，分隔，内容为(x_min,y_min,x_max,y_max,类别))
        x_min=int(info[0][0])
        y_min=int(info[0][1])
        x_max=x_min+int(info[0][2])
        y_max=y_min+int(info[0][3])
        
        box_info='%d,%d,%d,%d,%d'%(x_min,y_min,x_max,y_max,int(info[1]))
        f.write(box_info)
    f.write('\n')
f.close()
#验证集
name_box_id=defaultdict(list)
id_name=dict()
f=open(val_annotation_path,encoding='utf-8')
data=json.load(f)

annotations=data['annotatuons']

for ant in annotations:
    id=ant['image_id']
    name=os.path.join(val_dataset_path,'%012d.jpg'%id)
    cat=ant['category_id']
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    name_box_id[name].append([ant['bbox'],cat])
f=open(val_output_path,'w')
for key in name_box_id.keys():#读取不同图片的标签信息
    f.write(key)
    box_infos=name_box_id[key]
    for info in box_infos:#读取一张图中不同框的具体信息，以，分隔，内容为(x_min,y_min,x_max,y_max,类别))
        x_min=int(info[0][0])
        y_min=int(info[0][1])
        x_max=x_min+int(info[0][2])
        y_max=y_min+int(info[0][3])
        
        box_info='%d,%d,%d,%d,%d'%(x_min,y_min,x_max,y_max,int(info[1]))
        f.write(box_info)
    f.write('\n')
f.close()
