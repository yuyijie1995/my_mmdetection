import csv
import csv
import numpy as np
import os.path as osp
import shutil
import json
normal_path='/media/wrc/0EB90E450EB90E45/data/competition/classification/data_normal'
unnormal_path='/media/wrc/0EB90E450EB90E45/data/competition/classification/data_unnormal'
src_images_path='/media/wrc/0EB90C850EB90C85/kaggle_rsna_pneumonia_dancingbears-master/data/stage_2_train_images_png'
TrainPath='/media/wrc/0EB90E450EB90E45/data/比赛/'
img_path=TrainPath+'train'
label=TrainPath+'train_x-ray.json'
src_images_path2='/media/wrc/0EB90E450EB90E45/data/competition/train'

# with open('/media/wrc/0EB90C850EB90C85/kaggle_rsna_pneumonia_dancingbears-master/data/stage_2_train_labels.csv','r') as csvfile1:
#     reader = csv.reader(csvfile1)
#     rows1= [row for row in reader]
# with open('/media/wrc/0EB90C850EB90C85/kaggle_rsna_pneumonia_dancingbears-master/data/stage_2_detailed_class_info.csv','r') as csvfile2:
#     reader = csv.reader(csvfile2)
#     rows2= [row for row in reader]
#
# for info in rows2[1:]:
#
#     img_name=info[0]+'.png'
#     src_img_path=osp.join(src_images_path,img_name)
#     dst_img_path_normal=osp.join(normal_path,img_name)
#     dst_img_path_unnormal=osp.join(unnormal_path,img_name)
#     if info[1]=='Normal':
#         shutil.copy(src_img_path,dst_img_path_normal)
#     else:
#         shutil.copy(src_img_path,dst_img_path_unnormal)
#     print('completed one named %s'%img_name)

with open(label) as f:
    gt=json.load(f)
    for g in gt:
        src_image_path = osp.join(img_path, g['file_name'])
        if len(g['syms'])==0:
            dst_image_path=osp.join(normal_path,g['file_name'])
        else:
            dst_image_path = osp.join(unnormal_path, g['file_name'])
        shutil.copy(src_image_path,dst_image_path)


