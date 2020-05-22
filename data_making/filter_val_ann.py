import os
import shutil
###把VOC中train部分的图片和xml复制到duck中
image_path='/media/wrc/0EB90E450EB90E45/data/competition/train'
xml_path='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007/Annotations'
val_file='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007/ImageSets/Main/val.txt'
# dst_img='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duckval0.1/JPEGImages'
dst_img='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/JPEGImages'
dst_xml='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck519/Annotations'



f=open(val_file,'r')
for line in f.readlines():
    img_name=line.strip('\n')+'.png'
    xml_name=line.strip('\n')+'.xml'
    src_img_path=os.path.join(image_path,img_name)
    src_xml_path=os.path.join(xml_path,xml_name)
    dst_xml_path=os.path.join(dst_xml,xml_name)
    dst_img_path=os.path.join(dst_img,img_name)
    shutil.copy(src_img_path,dst_img_path)
    shutil.copy(src_xml_path,dst_xml_path)
    print('one file is completed')

