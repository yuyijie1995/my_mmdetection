# -*- coding: utf-8 -*-
#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET


START_BOUNDING_BOX_ID = 1

# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {
'Consolidation':1, 'Fibrosis':2, 'Effusion':3, 'Nodule':4, 'Mass':5,
        'Emphysema':6, 'Calcification':7, 'Atelectasis':8, 'Fracture':9
}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))

split_list=['train','val','test']

def convert(xml_path, xml_dir, json_path):

    for idx in split_list:
        xml_list=os.path.join(xml_path,idx+'.txt')

        list_fp = open(xml_list, 'r')
        json_dict = {"images":[], "type": "instances", "annotations": [],
                     "categories": []}
        categories = PRE_DEFINE_CATEGORIES
        bnd_id = START_BOUNDING_BOX_ID
        for line in list_fp:
            line = line.strip()

            print("Processing %s"%(line))
            xml_f = os.path.join(xml_dir, line+'.xml')
            tree = ET.parse(xml_f)
            root = tree.getroot()
            path = get(root, 'path')
            if len(path) == 1:
                filename = os.path.basename(path[0].text)
            elif len(path) == 0:
                filename = get_and_check(root, 'filename', 1).text
            else:
                raise NotImplementedError('%d paths found in %s'%(len(path), line))
            ## The filename must be a number
            image_id = get_filename_as_int(filename)
            # size = get_and_check(root, 'size', 1)
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)
            image = {'file_name': filename, 'height': height, 'width': width,
                     'id':image_id}
            json_dict['images'].append(image)
            ## Cruuently we do not support segmentation
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
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
                assert(xmax > xmin)
                assert(ymax > ymin)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                       image_id, 'bbox':[xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                       'segmentation': [[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]}
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

        for cate, cid in categories.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)

        if not os.path.exists(json_path):
            os.makedirs(json_path)
        json_file=os.path.join(json_path,idx+'.json')
        json_fp = open(json_file, 'w')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()
        list_fp.close()


if __name__ == '__main__':

    xml_list='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/ImageSets/Main/'
    xml_dir='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/Annotations'
    json_file='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/coco/annotations/'
    convert(xml_list, xml_dir, json_file)
    