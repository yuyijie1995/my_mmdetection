import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

#json2voc
CLASSES = ['Consolidation', 'Fibrosis', 'Effusion', 'Nodule', 'Mass',
        'Emphysema', 'Calcification', 'Atelectasis', 'Fracture']
# CLASSES = ['0', '1', '2', '3', '4',
#         '5', '6', '7', '8']
src_xml_dir = "./Annotations"

TrainPath='/media/wrc/0EB90E450EB90E45/data/比赛/'
img_path=TrainPath+'train'
label=TrainPath+'train_x-ray.json'

with open(label) as f:
    gt=json.load(f)
    print(gt)
    for g in gt:
        # img=cv2.imread(os.path.join(img_path,g['file_name']))
        img_name = g['file_name'][:-4]
        im = Image.open((img_path + '/' + img_name + '.png'))
        width, height = im.size
        xml_file = open((src_xml_dir + '/' + img_name + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str(img_name) + '.png' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')
        if len(g['syms'])==0:
            print('no_anno')
        else:
            for i,name in enumerate(g['syms']):
                x1=g['boxes'][i][0]
                y1=g['boxes'][i][1]
                x2=g['boxes'][i][2]
                y2=g['boxes'][i][3]
                xml_file.write('    <object>\n')
                # spt[0] = 'helmet'
                xml_file.write('        <name>' + name + '</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(x1) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(y1) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(x2) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(y2) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
        xml_file.write('</annotation>')

