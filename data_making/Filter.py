import os
import shutil
img_dir='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck/JPEGImages'
xml_dir='/media/wrc/0EB90E450EB90E45/data/competition/VOC2007_duck/Annotations'
img_lists=os.listdir(img_dir)
xml_lists=os.listdir(xml_dir)
for i in xml_lists:
    if i[:-4]+'.png' not in img_lists:
        os.remove(os.path.join(xml_dir,i))
