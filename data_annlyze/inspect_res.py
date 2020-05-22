import cv2
import json
import os
label='/home/wrc/Competition/mmdetection-master/val.json'
img_path='/media/wrc/0EB90E450EB90E45/data/competition/coco/val2017'
# model1_path='/home/wrc/Competition/mmdetection-master/res_submit_ensamble'
model2_path='/home/wrc/Competition/mmdetection-master/res_submit_newduck513_val'#绿色
model3_path='/home/wrc/Competition/mmdetection-master/517/duck517'#蓝色
model4_path='/home/wrc/Competition/mmdetection-master/res_submit_cas'#黄色
import csv

filename=[]
label_cla=[]
with open(os.path.join('./topk_ids_val.csv'), 'r') as out_file:
    reader = csv.reader(out_file)
    rows = [row for row in reader]
    class_dict=dict(rows)
acc=0

with open(label) as f:
    gt=json.load(f)
    for g in gt:
        img=cv2.imread(os.path.join(img_path,g['file_name']))
        model2_name=os.path.join(model2_path,g['file_name'][:-4]+'.json')
        model3_name=os.path.join(model3_path,g['file_name'][:-4]+'.json')
        model4_name=os.path.join(model4_path,g['file_name'][:-4]+'.json')
        f2=open(model2_name)
        f3=open(model3_name)
        f4=open(model4_name)
        model_bbox2=json.load(f2)
        model_bbox3=json.load(f3)
        model_bbox4=json.load(f4)
        if class_dict[g['file_name']] == '1' and len(g['syms']):
            acc+=1
        elif class_dict[g['file_name']]=='0' and len(g['syms'])==0:
            acc+=1
        cv2.putText(img,class_dict[g['file_name']], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 1)

        for i,name in enumerate(g['syms']):
            x1=g['boxes'][i][0]
            y1=g['boxes'][i][1]
            x2=g['boxes'][i][2]
            y2=g['boxes'][i][3]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, name, (x1 - 2, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        for bbox in model_bbox2:
            for k,v in bbox.items():
                x1=v[0]
                y1=v[1]
                x2=v[2]
                y2=v[3]
                label=k
                score=v[-1]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label += '|{:.02f}'.format(score)
                cv2.putText(img, label, (x1 - 2, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        for bbox in model_bbox3:
            for k,v in bbox.items():
                x1=v[0]
                y1=v[1]
                x2=v[2]
                y2=v[3]
                label=k
                score=v[-1]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                label += '|{:.02f}'.format(score)
                cv2.putText(img, label, (x1 - 2, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        for bbox in model_bbox4:
            for k,v in bbox.items():
                x1=v[0]
                y1=v[1]
                x2=v[2]
                y2=v[3]
                label=k
                score=v[-1]
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
                # label += '|{:.02f}'.format(score)
                # cv2.putText(img, label, (x1 - 2, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

        cv2.imwrite('vis_com2/%s'%g['file_name'],img)
        f2.close()
        f3.close()
        f4.close()
print(acc)