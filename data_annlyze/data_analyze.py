import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import copy

CLASSES = ['Consolidation', 'Fibrosis', 'Effusion', 'Nodule', 'Mass',
        'Emphysema', 'Calcification', 'Atelectasis', 'Fracture']
# CLASSES = ['0', '1', '2', '3', '4',
#         '5', '6', '7', '8']
bbox_sum=0
bbox_small=0
bbox_mid=0
bbox_big=0
img_sum=0
img_inclu_big=0
img_inclu_mid=0
img_inclu_small=0
w_h_ratio={'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
TrainPath='/media/wrc/0EB90E450EB90E45/data/比赛/'
img_path=TrainPath+'train'
label=TrainPath+'train_x-ray.json'
classes_count={}
for i,name in enumerate(CLASSES):
    classes_count[name]=0
max_width_height=0
min_width_height=float('inf')



def statistic_obj_area(area):
    global bbox_small
    global bbox_big
    global bbox_mid
    if area <= 32 * 32:
        bbox_small += 1
    elif area < 96 * 96:
        bbox_mid += 1
    else:
        bbox_big += 1

def statistic_h_w_ratio(w_h_):
    global w_h_ratio
    if w_h_ <= 1.5:
        w_h_ratio['1'] += 1
    elif w_h_ <= 2.5:
        w_h_ratio['2'] += 1

    elif w_h_ <= 3.5:
        w_h_ratio['3'] += 1
    elif w_h_ <= 4.5:
        w_h_ratio['4'] += 1
    elif w_h_ <= 5.5:
        w_h_ratio['5'] += 1
    elif w_h_ <= 6.5:
        w_h_ratio['6'] += 1

    elif w_h_ <= 7.5:
        w_h_ratio['7'] += 1
    elif w_h_ <= 8.5:
        w_h_ratio['8'] += 1

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))


vis_img_flag=False


min_area=float('inf')

with open(label) as f:
    gt=json.load(f)
    for g in gt:
        img_sum+=1
        img=cv2.imread(os.path.join(img_path,g['file_name']))
        if len(g['syms'])==0:
            continue
        else:
            o_small = copy.copy(bbox_small)
            o_mid = copy.copy(bbox_mid)
            o_big = copy.copy(bbox_big)
            for i,name in enumerate(g['syms']):
                classes_count[name]+=1
                bbox_sum+=1
                x1=g['boxes'][i][0]
                y1=g['boxes'][i][1]
                x2=g['boxes'][i][2]
                y2=g['boxes'][i][3]
                width=x2-x1
                height=y2-y1
                area=height*width
                min_area=area if area<min_area else min_area
                # statistic_obj_area(area)
                # if height>width:
                #     width,height=height,width#始终保证width大于height
                # max_width_height=width/height if width/height>max_width_height else max_width_height
                # min_width_height=width/height if width/height<min_width_height else min_width_height
                # w_h_=width/height
                # statistic_h_w_ratio(w_h_)
                # if vis_img_flag:
                #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                #
                #     cv2.putText(img, name, (x1 - 2, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
            # if o_big!=bbox_big:
            #     img_inclu_big+=1
            # if o_mid!=bbox_mid:
            #     img_inclu_mid+=1
            # if o_small!=bbox_small:
            #     img_inclu_small+=1
        print('finished one')
            # saved_name=os.path.join('./vised_imgs',g['file_name'])
            # cv2.imwrite(saved_name,img)
            # print('a file {} is completed!'.format(g['file_name']))
print('min_area= ',min_area)
print('w_h_ratio:')
print(w_h_ratio)
print('包含各种大小目标的图片占比:')
print((round(img_inclu_small/img_sum,2),round(img_inclu_mid/img_sum,2),round(img_inclu_big/img_sum,2)))
print('大中小目标占比:')
print((round(bbox_small/bbox_sum,2),round(bbox_mid/bbox_sum,2),round(bbox_big/bbox_sum,2)))
# name_list=classes_count.keys()
# class_count_new={
# 'Consolidation':'0', 'Fibrosis':'1', 'Effusion':'2', 'Nodule':'3', 'Mass':'4',
#         'Emphysema':'5', 'Calcification':'6', 'Atelectasis':'7', 'Fracture':'8'
# }
# val_list=classes_count.values()
# name_new_list=[]
# for name in name_list:
#     name_new_list.append(class_count_new[name])
# a=plt.bar(range(len(name_list)),val_list,tick_label=name_new_list)
# autolabel(a)
# plt.xticks(range(len(name_list)),name_list,rotation=30)
# plt.savefig('class_analyze.png')
# plt.show()
### 统计大中小三类目标的数量
### 统计目标长宽比