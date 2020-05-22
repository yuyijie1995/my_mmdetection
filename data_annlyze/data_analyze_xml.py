#过滤duck出来的没有标签的数据
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import cv2
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
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))

def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))
classes_count={}

CLASSES = ['Consolidation', 'Fibrosis', 'Effusion', 'Nodule', 'Mass',
        'Emphysema', 'Calcification', 'Atelectasis', 'Fracture']
for i,name in enumerate(CLASSES):
    classes_count[name]=0

src_xml_path="/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck516/Annotations"
src_img_path="/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck516/JPEGImages"
xml_lists=os.listdir(src_xml_path)
for xml_file in xml_lists:
    img_file=xml_file[:-4]+'.png'
    xml_dir=os.path.join(src_xml_path,xml_file)
    img_dir=os.path.join(src_img_path,img_file)
    img = cv2.imread(img_dir)
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    rs=root.findall('object')
    for obj in get(root, 'object'):
        category = get_and_check(obj, 'name', 1).text
        bndbox = get_and_check(obj, 'bndbox', 1)

        # print(get_and_check(bndbox, 'xmin', 1).text )
        xmin = int(float(get_and_check(bndbox, 'xmin', 1).text)) - 1
        ymin = int(float(get_and_check(bndbox, 'ymin', 1).text)) - 1
        xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
        ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
        classes_count[category]+=1
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        cv2.putText(img, category, (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.imwrite('vis_duck/%s' % img_file, img)

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
# plt.savefig('class_analyze_duck.png')
# plt.show()