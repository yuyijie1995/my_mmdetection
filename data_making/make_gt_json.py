import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


CLASSES = ['Consolidation', 'Fibrosis', 'Effusion', 'Nodule', 'Mass',
        'Emphysema', 'Calcification', 'Atelectasis', 'Fracture']

split_list=['val','test']
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
def convert(xml_path, xml_dir, json_path):

    for idx in split_list:
        xml_list=os.path.join(xml_path,idx+'.txt')
        json_file=os.path.join(json_path,idx+'.json')

        json_fp=open(json_file,'w')
        data_list=[]
        list_fp = open(xml_list, 'r')
        for line in list_fp:
            line = line.strip()
            json_dict = {"file_name": '', "syms": [], "boxes": []}

            print("Processing %s"%(line))
            xml_f = os.path.join(xml_dir, line+'.xml')
            tree = ET.parse(xml_f)
            root = tree.getroot()



            json_dict['file_name']=line+'.png'
            ## Cruuently we do not support segmentation
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                json_dict['syms'].append(category)
                bndbox = get_and_check(obj, 'bndbox', 1)

                # print(get_and_check(bndbox, 'xmin', 1).text )
                xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
                ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
                xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
                ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))

                # print(xmax)
                # print(xmin)
                assert(xmax > xmin)
                assert(ymax > ymin)
                json_dict['boxes'].append([xmin,ymin,xmax,ymax])

            data_list.append(json_dict)

        if not os.path.exists(json_path):
            os.makedirs(json_path)
        json_str = json.dumps(data_list)
        json_fp.write(json_str)
        json_fp.close()



if __name__ == '__main__':

    xml_list='/media/wrc/0EB90E450EB90E45/data/competition/ImageSets/Main/'
    xml_dir='/media/wrc/0EB90E450EB90E45/data/competition/Annotations'
    json_file='.'
    convert(xml_list, xml_dir, json_file)
