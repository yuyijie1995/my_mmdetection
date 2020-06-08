import os
import cv2
import json
import matplotlib.pyplot as plt
from PIL import ImageDraw
### 显示各个类别在图中的位置分布
from PIL import Image
crop_image_path='/media/wrc/0EB90E450EB90E45/data/比赛/vis_category'
IMAGE_SAVE_PATH='/media/wrc/0EB90E450EB90E45/data/比赛'
CLASSES = ['Consolidation', 'Fibrosis', 'Effusion', 'Nodule', 'Mass',
        'Emphysema', 'Calcification', 'Atelectasis', 'Fracture']
# CLASSES = ['0', '1', '2', '3', '4',
#         '5', '6', '7', '8']
bbox_sum=0
bbox_small=0
bbox_mid=0
bbox_big=0
img_sum=0
img_inclu_big=0
img_inclu_mid=0
img_inclu_small=0
TrainPath='/media/wrc/0EB90E450EB90E45/data/比赛/'
img_path=TrainPath+'train'
label=TrainPath+'train_x-ray.json'
classes_count={}
for i,name in enumerate(CLASSES):
    classes_count[name]=0
max_width_height=0
min_width_height=float('inf')

save_image1=Image.new('RGB', (1024, 1024))  # 创建一个新图
save_image2=Image.new('RGB', (1024, 1024))  # 创建一个新图
save_image3=Image.new('RGB', (1024, 1024))  # 创建一个新图
save_image4=Image.new('RGB', (1024, 1024))  # 创建一个新图
save_image5=Image.new('RGB', (1024, 1024))  # 创建一个新图
save_image6=Image.new('RGB', (1024, 1024))  # 创建一个新图
save_image7=Image.new('RGB', (1024, 1024))  # 创建一个新图
save_image8=Image.new('RGB', (1024, 1024))  # 创建一个新图
save_image9=Image.new('RGB', (1024, 1024))  # 创建一个新图
draw1 = ImageDraw.Draw(save_image1)
draw2 = ImageDraw.Draw(save_image2)
draw3 = ImageDraw.Draw(save_image3)
draw4 = ImageDraw.Draw(save_image4)
draw5 = ImageDraw.Draw(save_image5)
draw6 = ImageDraw.Draw(save_image6)
draw7 = ImageDraw.Draw(save_image7)
draw8 = ImageDraw.Draw(save_image8)
draw9 = ImageDraw.Draw(save_image9)



crop_count=0
with open(label) as f:
    gt=json.load(f)
    for g in gt:
        img_sum+=1
        img=cv2.imread(os.path.join(img_path,g['file_name']))
        if len(g['syms'])==0:
            continue
        else:
            for i,name in enumerate(g['syms']):
                x1=g['boxes'][i][0]
                y1=g['boxes'][i][1]
                x2=g['boxes'][i][2]
                y2=g['boxes'][i][3]

                if name=='Consolidation':
                    draw1.rectangle([x1, y1, x2, y2], outline='red')
                if name=='Fibrosis':
                    draw2.rectangle([x1, y1, x2, y2], outline='red')
                if name=='Effusion':
                    draw3.rectangle([x1, y1, x2, y2], outline='red')
                if name=='Nodule':
                    draw4.rectangle([x1, y1, x2, y2], outline='red')
                if name=='Mass':
                    draw5.rectangle([x1, y1, x2, y2], outline='red')
                if name=='Emphysema':
                    draw6.rectangle([x1, y1, x2, y2], outline='red')
                if name=='Calcification':
                    draw7.rectangle([x1, y1, x2, y2], outline='red')
                    if (x2-x1)*(y2-y1)>200*200:
                        print(g['file_name'])
                if name=='Atelectasis':
                    draw8.rectangle([x1, y1, x2, y2], outline='red')
                if name=='Fracture':
                    draw9.rectangle([x1, y1, x2, y2], outline='red')
                # cropped = img[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]
                # img_save_crop=os.path.join(crop_image_path,'%s'%name)
                # if not os.path.exists(img_save_crop):
                #     os.makedirs(img_save_crop)
                # img_crop_name=str(crop_count)+'.png'
                # img_crop_path=os.path.join(img_save_crop,img_crop_name)
                # cv2.imwrite(img_crop_path, cropped)
                crop_count+=1
    save_image1.save(IMAGE_SAVE_PATH + '/' + 'distri1', format='png')
    save_image2.save(IMAGE_SAVE_PATH + '/' + 'distri2', format='png')
    save_image3.save(IMAGE_SAVE_PATH + '/' + 'distri3', format='png')
    save_image4.save(IMAGE_SAVE_PATH + '/' + 'distri4', format='png')
    save_image5.save(IMAGE_SAVE_PATH + '/' + 'distri5', format='png')
    save_image6.save(IMAGE_SAVE_PATH + '/' + 'distri6', format='png')
    save_image7.save(IMAGE_SAVE_PATH + '/' + 'distri7', format='png')
    save_image8.save(IMAGE_SAVE_PATH + '/' + 'distri8', format='png')
    save_image9.save(IMAGE_SAVE_PATH + '/' + 'distri9', format='png')
