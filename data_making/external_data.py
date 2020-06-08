import os
import csv
import numpy as np
from PIL import Image
import shutil
#通过csv文件制作xml文件
ann_path='/media/wrc/0EB90C850EB90C85/Data_Entry_2017_v2020.csv'
image_path='/media/wrc/0EB90C850EB90C85/images'
CLASSES = ['Consolidation', 'Fibrosis', 'Effusion', 'Nodule', 'Mass',
        'Emphysema', 'Calcification', 'Atelectasis', 'Fracture']
dst_img_path='/media/wrc/0EB90C850EB90C85/voc_ex/JPEGImages'
dst_xml_path='/media/wrc/0EB90C850EB90C85/voc_ex/Annotations'

dst_path='/media/wrc/0EB90E450EB90E45/data/competition/classification2'
dst_path_template='/media/wrc/0EB90E450EB90E45/data/competition/duck_514/template'


if not os.path.exists(dst_xml_path):
    os.makedirs(dst_xml_path)
if not os.path.exists(dst_img_path):
    os.makedirs(dst_img_path)
with open(ann_path,'r') as csvfile1:
    reader = csv.reader(csvfile1)
    rows1= [row for row in reader]
    print(rows1)
    for g in rows1[1:]:
        src_image=os.path.join(image_path,g[0])
        # dst_image0 = os.path.join(dst_path, '0',g[0])#0 是正常
        # dst_image1 = os.path.join(dst_path, '1',g[0])#1是不正常
        dst_image_temp=os.path.join(dst_path_template,g[0])
        if g[1]=='No Finding':
            shutil.copy(src_image,dst_image_temp)
        # elif g[1] not in CLASSES:
        #     continue
        # else:
        #     shutil.copy(src_image,dst_image1)
        print('img finished!')



# with open(ann_path,'r') as csvfile1:
#     reader = csv.reader(csvfile1)
#     rows1= [row for row in reader]
#     print(rows1)
#     for g in rows1:
#         if g[1] in CLASSES:
#             try:
#                 img_src=os.path.join(image_path,g[0])
#                 img_normal_name=g[0][:-4]
#                 img_name = g[0].replace('_','')[:-4]
#                 img_dst=os.path.join(dst_img_path,img_name+'.png')
#                 shutil.copy(img_src,img_dst)
#                 im = Image.open((image_path + '/' + img_normal_name + '.png'))
#                 width, height = im.size
#                 xml_file = open((dst_xml_path + '/' + img_name + '.xml'), 'w')
#                 xml_file.write('<annotation>\n')
#                 xml_file.write('    <folder>VOC2007</folder>\n')
#                 xml_file.write('    <filename>' + str(img_name) + '.png' + '</filename>\n')
#                 xml_file.write('    <size>\n')
#                 xml_file.write('        <width>' + str(width) + '</width>\n')
#                 xml_file.write('        <height>' + str(height) + '</height>\n')
#                 xml_file.write('        <depth>3</depth>\n')
#                 xml_file.write('    </size>\n')
#                 x1=int(float(g[2]))
#                 y1=int(float(g[3]))
#                 w=int(float(g[4]))
#                 h=int(float(g[5]))
#                 x2=x1+w
#                 y2=y1+h
#                 xml_file.write('    <object>\n')
#                 # spt[0] = 'helmet'
#                 xml_file.write('        <name>' + g[1] + '</name>\n')
#                 xml_file.write('        <pose>Unspecified</pose>\n')
#                 xml_file.write('        <truncated>0</truncated>\n')
#                 xml_file.write('        <difficult>0</difficult>\n')
#                 xml_file.write('        <bndbox>\n')
#                 xml_file.write('            <xmin>' + str(x1) + '</xmin>\n')
#                 xml_file.write('            <ymin>' + str(y1) + '</ymin>\n')
#                 xml_file.write('            <xmax>' + str(x2) + '</xmax>\n')
#                 xml_file.write('            <ymax>' + str(y2) + '</ymax>\n')
#                 xml_file.write('        </bndbox>\n')
#                 xml_file.write('    </object>\n')
#                 xml_file.write('</annotation>')
#             except FileNotFoundError:
#                 print('one file not found')