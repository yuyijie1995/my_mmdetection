# create_train_test_txt.py
# encoding:utf-8
import pdb
import glob
import os
import random
import math
import shutil

img0_list_path = os.listdir('/media/wrc/0EB90E450EB90E45/data/competition/classification2/0')
img1_list_path = os.listdir('/media/wrc/0EB90E450EB90E45/data/competition/classification2/1')

dst_img_0_path='/media/wrc/0EB90E450EB90E45/data/competition/classification2/train/0'
dst_img_1_path='/media/wrc/0EB90E450EB90E45/data/competition/classification2/train/1'
dst_img_0_path_val='/media/wrc/0EB90E450EB90E45/data/competition/classification2/val/0'
dst_img_1_path_val='/media/wrc/0EB90E450EB90E45/data/competition/classification2/val/1'

src_img_path0='/media/wrc/0EB90E450EB90E45/data/competition/classification2/0'
src_img_path1='/media/wrc/0EB90E450EB90E45/data/competition/classification2/1'

txt_list0 = []
txt_list1 = []

for item in img0_list_path:

    txt_list0.append(item)

for item in img1_list_path:
    txt_list1.append(item)


num_trainval0 = txt_list0

num_trainval1 = txt_list1

num_train0 = random.sample(num_trainval0, math.floor(len(num_trainval0) * 9 / 10.0))  # 可修改百分比
num_train1 = random.sample(num_trainval1, math.floor(len(num_trainval1) * 9 / 10.0))  # 可修改百分比

for img in num_train0:

    src_img0=os.path.join(src_img_path0,img)
    dst_img0=os.path.join(dst_img_0_path,img)
    shutil.copy(src_img0,dst_img0)

    print('finish 1 train')
for img in num_train1:
    src_img1 = os.path.join(src_img_path1, img)
    dst_img1 = os.path.join(dst_img_1_path, img)
    shutil.copy(src_img1, dst_img1)


num_val0 = list(set(num_trainval0).difference(set(num_train0)))
num_val1 = list(set(num_trainval1).difference(set(num_train1)))
for img in num_val0:
    src_img0 = os.path.join(src_img_path0, img)
    dst_img0 = os.path.join(dst_img_0_path_val, img)
    shutil.copy(src_img0, dst_img0)
for img in num_val1:
    src_img1 = os.path.join(src_img_path1, img)
    dst_img1 = os.path.join(dst_img_1_path_val, img)
    shutil.copy(src_img1, dst_img1)
    print('finish 1val')

