import os
import json
import numpy as np
import pandas as pd
import cv2
import glob
import random
from PIL import Image
import time
from sklearn import metrics as mr
import shutil
import sys
from pycocotools.coco import COCO
#通过coco文件制作xml文件
random.seed(2019)
CLASSES = ['Consolidation', 'Fibrosis', 'Effusion', 'Nodule', 'Mass',
        'Emphysema', 'Calcification', 'Atelectasis', 'Fracture']

# src_xml_dir = "/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duckval0.1/Annotations"
src_xml_dir = "/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/Annotations"
if not os.path.exists(src_xml_dir):
    os.makedirs(src_xml_dir)
# save_dir='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duckval0.1/JPEGImages'
save_dir='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/JPEGImages'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# template_dir='/media/wrc/0EB90E450EB90E45/data/competition/duck_514/template'
template_dir='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duckval0.1/template'
if not os.path.exists(template_dir):
    os.makedirs(template_dir)

TrainPath='/media/wrc/0EB90E450EB90E45/data/competition/coco'
# TrainPath='/media/wrc/0EB90C850EB90C85/voc_ex/coco_ex_duck'
# img_path=TrainPath+'/train2017'
img_path=TrainPath+'/val2017'
# img_path=TrainPath+'/trainset'
# label=TrainPath+'/annotations/instances_train2017.json'
label=TrainPath+'/annotations/instances_val2017.json'
# label=TrainPath+'/annotations/train.json'


img_name_count=0
ring_width=10#调整为10 是不是更为合理
RGBA_num=0
template_imgs=[os.path.join(template_dir,img) for img in os.listdir(template_dir)]
mask = cv2.imread('mask.jpg', 0)
coco=COCO(label)
cat_ids=coco.getAnnIds()
cat2label={cat_id: i + 1
            for i, cat_id in enumerate(cat_ids)}
img_ids = coco.getImgIds()
img_infos = []
for i in img_ids:
    info = coco.loadImgs([i])[0]
    info['filename'] = info['file_name']
    img_infos.append(info)
print(cat_ids)

for img_info in img_infos:
    img_id=img_info['id']
    ann_ids=coco.getAnnIds(imgIds=[img_id])
    ann_info=coco.loadAnns(ann_ids)
    img_name=img_info['file_name']
    im_test = Image.open((img_path + '/' + img_name))
    im_template = Image.open(random.choice(template_imgs))
    save_template_name = '0' + img_name
    width, height = im_test.size
    width_t, height_t = im_template.size
    img_name = img_info['file_name'][:-4]
    xml_file = open((src_xml_dir + '/' + '0' + img_name + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + str(save_template_name[:-4]) + '.png' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width_t) + '</width>\n')
    xml_file.write('        <height>' + str(height_t) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        if ann['area'] <= 0 or w < 1 or h < 1:
            continue
        bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
        x2=x1+w-1
        y2=y1+h-1
        h = abs(x2 - x1)
        w = abs(y2 - y1)
        w_h = round(w / h, 2)
        h_w = round(h / w, 2)
        left_top_x = random.randint(1, im_test.size[0])
        left_top_y = random.randint(1, im_test.size[1])
        label=CLASSES[cat2label[ann['category_id']]-1]
        mask[int(left_top_y - ring_width):int(left_top_y + w + ring_width),
        int(left_top_x - ring_width):int(left_top_x + h + ring_width)] = 255
        mask[int(left_top_y):int(left_top_y + w), int(left_top_x):int(left_top_x + h)] = 0
        patch = im_test.crop((x1, y1, x2, y2))
        patch1 = patch.copy()
        patch2 = im_template.crop(
            (left_top_x, left_top_y, int(left_top_x + patch1.size[0]), int(left_top_y + patch1.size[1])))
        if patch2.mode != 'L':
            patch2 = patch2.convert('L')
        if patch2.mode == 'RGBA':
            RGBA_num += 1
            patch2 = patch2.convert('RGB')
            patch2 = patch2.convert('L')
        # print('bbox:', (left_top_x, left_top_y, int(left_top_x + (x2 - x1)), int(left_top_y + (y2 - y1))))
        pat1 = patch1.copy()
        pat2 = patch2.copy()
        patch1 = np.resize(patch1, -1)
        patch2 = np.resize(patch2, -1)
        # 相似度检测
        try:
            mutual_infor = mr.mutual_info_score(patch1, patch2)
        except ValueError:
            pass
        print(mutual_infor)
        if mutual_infor > 0.8:
            im_template.paste(patch, (left_top_x, left_top_y))
            im_template = cv2.cvtColor(np.asarray(im_template), cv2.COLOR_RGB2BGR)

            im_template = cv2.inpaint(im_template, mask, 3, cv2.INPAINT_TELEA)
            im_template = Image.fromarray(cv2.cvtColor(im_template, cv2.COLOR_BGR2RGB))

            xml_file.write('    <object>\n')
            # spt[0] = 'helmet'
            xml_file.write('        <name>' + label + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(left_top_x) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(left_top_y) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(left_top_x + h) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(left_top_y + w) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        else:
            continue
    im_template.save(save_dir + '/' + save_template_name)
    img_name_count += 1
    print('img_%s is completed' % img_name)
    xml_file.write('</annotation>')


