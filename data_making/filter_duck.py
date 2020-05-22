#过滤duck出来的没有标签的数据
import xml.etree.ElementTree as ET
import os



src_xml_path="/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck518/Annotations"
src_img_path="/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck518/JPEGImages"
xml_lists=os.listdir(src_xml_path)
for xml_file in xml_lists:
    img_file=xml_file[:-4]+'.png'
    xml_dir=os.path.join(src_xml_path,xml_file)
    img_dir=os.path.join(src_img_path,img_file)
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    rs=root.findall('object')
    if len(rs)==0:
        os.remove(xml_dir)
        os.remove(img_dir)
        print('img %s is deleted!'%img_file)