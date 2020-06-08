import cv2
import os
# opencv读取图像 颜色取反
img_path='/media/wrc/0EB90E450EB90E45/data/competition/coco/train2017'
dst_path='/media/wrc/0EB90E450EB90E45/data/competition/coco/train2017_reverse'
img_lists=os.listdir(img_path)
for imgname in img_lists:
    img_dir=os.path.join(img_path,imgname)
    dst_img_dir=os.path.join(dst_path,imgname)

    img=cv2.imread(img_dir,1)
    img_shape = img.shape  # 图像大小(565, 650, 3)
    h = img_shape[0]
    w = img_shape[1]
    # 最大图像灰度值减去原图像，即可得到反转的图像
    dst = 255 - img
    cv2.imwrite(dst_img_dir,dst)

