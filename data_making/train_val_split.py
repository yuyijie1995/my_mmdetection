# create_train_test_txt.py
# encoding:utf-8
import pdb
import glob
import os
import random
import math


# txt_list_path = glob.glob('/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duckval0.1/Annotations/*.xml')
txt_list_path = glob.glob('/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/Annotations/*.xml')
txt_list = []

for item in txt_list_path:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    txt_list.append(temp1)
txt_list.sort()
print(txt_list, end='\n\n')

### 均匀划分
#
# num_trainval1 = random.sample(txt_list, math.floor(len(txt_list) * 1/ 10.0))
# num_last1=list(set(txt_list).difference(num_trainval1))
# num_last1.sort()
# print(len(num_last1), end='\n')
#
# num_trainval2 = random.sample(num_last1, math.floor(len(num_last1) * 1/ 10.0))
# num_last2=list(set(num_last1).difference(num_trainval2))
# num_last2.sort()
# print(len(num_last2), end='\n')
#
# num_trainval3 = random.sample(num_last2, math.floor(len(num_last2) * 1/ 10.0))
# num_last3=list(set(num_last2).difference(num_trainval3))
# num_last3.sort()
# print(len(num_last3), end='\n')
#
# num_trainval4 = random.sample(num_last3, math.floor(len(num_last3) * 1/ 10.0))
# num_last4=list(set(num_last3).difference(num_trainval4))
# num_last4.sort()
# print(len(num_last4), end='\n')
#
# num_trainval5 = random.sample(num_last4, math.floor(len(num_last4) * 1/ 10.0))
# num_last5=list(set(num_last4).difference(num_trainval5))
# num_last5.sort()
# print(len(num_last5), end='\n')
#
# num_trainval6 = random.sample(num_last5, math.floor(len(num_last5) * 1/ 10.0))
# num_last6=list(set(num_last5).difference(num_trainval6))
# num_last6.sort()
# print(len(num_last6), end='\n')
#
# num_trainval7 = random.sample(num_last6, math.floor(len(num_last6) * 1/ 10.0))
# num_last7=list(set(num_last6).difference(num_trainval7))
# num_last7.sort()
# print(len(num_last7), end='\n')
#
# num_trainval8 = random.sample(num_last7, math.floor(len(num_last7) * 1/ 10.0))
# num_last8=list(set(num_last7).difference(num_trainval8))
# num_last8.sort()
# print(len(num_last8), end='\n')
#
# num_trainval9 = random.sample(num_last8, math.floor(len(num_last8) * 1/ 10.0))
# num_last9=list(set(num_last8).difference(num_trainval9))
# num_last9.sort()
# print(len(num_last9), end='\n')


# 有博客建议train:val:test=8:1:1，先尝试用一下
num_trainval = random.sample(txt_list, math.floor(len(txt_list) * 10/ 10.0))  # 可修改百分比
num_trainval.sort()
print(num_trainval, end='\n\n')

num_train = random.sample(num_trainval, math.floor(len(num_trainval) * 10 / 10.0))  # 可修改百分比
num_train.sort()
print(num_train, end='\n\n')

num_val = list(set(num_trainval).difference(set(num_train)))
num_val.sort()
print(num_val, end='\n\n')

num_test = list(set(txt_list).difference(set(num_trainval)))
num_test.sort()
print(num_test, end='\n\n')

# pdb.set_trace()

Main_path = '/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/ImageSets/Main/'
test_Main_path = '/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/ImageSets/Main/'

if not os.path.exists(Main_path):
    os.makedirs(Main_path)
if not os.path.exists(test_Main_path):
    os.makedirs(test_Main_path)
train_test_name = ['trainval', 'train', 'val', 'test']
# category_name = ['helmet']

# 循环写trainvl train val test
for item_train_test_name in train_test_name:
    list_name = 'num_'
    list_name += item_train_test_name
    # train_test_txt_name = Main_path + item_train_test_name + '.txt'
    train_test_txt_name = test_Main_path + item_train_test_name + '.txt'
    try:
        # 写单个文件
        with open(train_test_txt_name, 'w') as w_tdf:
            # 一行一行写
            for item in eval(list_name):
                w_tdf.write(item + '\n')
    except IOError as ioerr:
        print('File error:' + str(ioerr))

