from PIL import Image
import numpy as np
import json
import os
from PIL import ImageDraw
import cv2
images_path='/media/wrc/0EB90E450EB90E45/data/比赛/train'
label='/media/wrc/0EB90E450EB90E45/data/比赛/train_x-ray.json'
IMAGE_SAVE_PATH='.'
col=2
row=2

#mosaic数据拼接方法的效果验证

img_info={}
img_count=0
with open(label) as f:
    gt=json.load(f)
    for g in gt:
        if img_count>=4:
            break
        img=Image.open(os.path.join(images_path,g['file_name']))
        if len(g['syms'])==0:
            continue
        else:
            bbox=[]
            for i,name in enumerate(g['syms']):
                x1=g['boxes'][i][0]
                y1=g['boxes'][i][1]
                x2=g['boxes'][i][2]
                y2=g['boxes'][i][3]
                bbox.append([x1,y1,x2,y2])
            img_info[g['file_name']]=bbox
            img_count+=1
print(img_info)
img_size=1024

def image_compose2(imgs,bboxes_lists,imgname):
    #传入的应该是原始比例的图像
    for bboxes in bbox_lists:
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, 512 - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, 512 - 1)
    save_image=Image.new('RGB', (col * img_size//2, row * img_size//2))  # 创建一个新图
    for i in range(row):
        for j in range(col):
            save_image.paste(imgs[i*row+j], (j* img_size//2, i * img_size//2))
            draw=ImageDraw.Draw(save_image)
            for bbox in bboxes_lists[i*row+j]:
                draw.rectangle([bbox[0]+(j*img_size//2),bbox[1]+(i*img_size//2),bbox[2]+(j*img_size//2),bbox[3]+(i*img_size//2)],outline='red')

    save_image.save(IMAGE_SAVE_PATH+'/'+imgname,format='png')

img_lists=[]
bbox_lists=[]
target_scale=0.5

for k,v in img_info.items():
    img=Image.open(os.path.join(images_path,k)).resize((int(img_size * target_scale), int(img_size * target_scale)), Image.ANTIALIAS)
    img_normal=Image.open(os.path.join(images_path,k))
    img_lists.append(img)
    bbox_nd=np.array(v)*0.5
    # v=bbox_nd.tolist()
    bbox_lists.append(bbox_nd)

    # img_cv=np.array(img)
    draw = ImageDraw.Draw(img_normal)

    for box in v:
        draw.rectangle([box[0], box[1], box[2], box[3]],outline='red')
    # cv2.imwrite('%s'%k,img_cv)
    img_normal.save('%s'%k)
# bboxes=np.array(bbox_lists)
image_compose2(img_lists,bbox_lists,'23.png')
