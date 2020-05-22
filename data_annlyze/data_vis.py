# -*- coding: utf-8 -*-
# !/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET
import cv2
START_BOUNDING_BOX_ID = 1

# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {
    'Consolidation': 1, 'Fibrosis': 2, 'Effusion': 3, 'Nodule': 4, 'Mass': 5,
    'Emphysema': 6, 'Calcification': 7, 'Atelectasis': 8, 'Fracture': 9
}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


split_list = ['train', 'val', 'test']


def vis( xml_dir):


    # img_dir=xml_dir[:-4]+'imgs'
    img_dir='/media/wrc/0EB90C850EB90C85/voc_ex/JPEGImages'
    xml_lists=os.listdir(xml_dir)
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in xml_lists:
        line = line.strip()
        file_name=line[:-4]

        print("Processing %s" % (line))
        xml_f = os.path.join(xml_dir, file_name + '.xml')
        img_f = os.path.join(img_dir, file_name + '.png')
        img=cv2.imread(img_dir+'/'+file_name+'.png')
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s' % (len(path), line))
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)

            # print(get_and_check(bndbox, 'xmin', 1).text )
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text)) - 1
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text)) - 1
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            # print(xmax)
            # print(xmin)
            assert (xmax > xmin)
            assert (ymax > ymin)
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),4)
            cv2.putText(img, category, (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
        # save_dir="/media/wrc/0EB90E450EB90E45/data/competition/duck_injection/save_vis/{}".format(file_name)
        save_dir="/media/wrc/0EB90C850EB90C85/voc_ex/save_vis/{}".format(file_name)
        cv2.imwrite('{}.png'.format(save_dir),img)






if __name__ == '__main__':
    # xml_dir = "/media/wrc/0EB90E450EB90E45/data/competition/duck_injection/save/xmls"
    xml_dir = "/media/wrc/0EB90C850EB90C85/voc_ex/Annotations"
    vis( xml_dir)
