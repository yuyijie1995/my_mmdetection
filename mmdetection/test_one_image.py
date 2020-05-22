from mmdet.apis import init_detector, inference_detector, show_result
import os
import argparse
import sys
import mmcv
import numpy as np
import torch
import pycocotools.mask as maskUtils
import cv2
from mmcv.image import imread,imwrite
import json

import time
txt_open=True

os.environ['CUDA_VISIBLE_DEVICES']='0'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj),1)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# print('config '/media/wrc/0EB90E450EB90E45/mmdetection/configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x.py' checkpoint '/media/wrc/0EB90E450EB90E45/mmdetection/work_dirs/ga_faster_rcnn_r50_caffe_full_model_fpn_1x/latest.pth' ')
def parse_args():
    parser = argparse.ArgumentParser(description='in and out imgs')
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/faster_rcnn_mdconv_c3-c5_group4_r50_fpn_1x.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/ga_retinanet_x101_32x4d_fpn_1x.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/faster_rcnn_mdconv_c3-c5_group4_r50_fpn_1x_mixup.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/new_duck513.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/add_context.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/duck516_mix.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/duck518.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/duck513_weighted.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/faster_100.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/faster_100mix.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/faster_100_mosaic.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/cas_final.py', type=str)
    parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/add_context_final.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/sepc_freeanchor.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/Masaic.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/mixup_bigger.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/cutout_test.py', type=str)
    # parser.add_argument('--config', dest='config',help='config_file',default='/home/wrc/Competition/mmdetection-master/configs/FeiYan/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/faster_rcnn_r50_fpn_1x_mdconv_g_multiscale_OHEM2_mixup/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/mixup_duck513/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/add_context515/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/duck516mix/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/mosaic515/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/mixup_duck513_weighted/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/final_nomix/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/final_mix/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/final_mosaic/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/final_cas/latest.pth', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/add_context_final/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/retinanet_free_anchor_r50_fpn_1x_100/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/mixup_bigger/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/faster_rcnn_r50_fpn_1x_mdconv_g_multiscale_OHEM2/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/data/competition/work_dirs/ga_retinanet_bigger/latest.pth', type=str)
    # parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/home/wrc/Competition/mmdetection-master/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/latest.pth', type=str)

    # if len(sys.argv) <= 1:
    #     parser.print_help()
    #     sys.exit(1)
    args = parser.parse_args()
    return args

CATEGORIES = {
    0:'Consolidation',1: 'Fibrosis', 2:'Effusion', 3: 'Nodule', 4:'Mass',
    5:'Emphysema', 6:'Calcification', 7:'Atelectasis', 8:'Fracture'
}



def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    model = init_detector(config_file, checkpoint_file)
    print(model.CLASSES)

    # test_data_root = '/media/wrc/0EB90E450EB90E45/data/competition/coco/val2017'
    test_data_root = '/media/wrc/0EB90E450EB90E45/data/competition/A'



    # savedir = './saved_images_submit_final/'
    # savedir = './saved_images_submit_mix_cutout200/'
    # if not os.path.exists(savedir):
    #     os.mkdir(savedir)
    score_thr=0.25#给大图不同的thresh
    # assert isinstance(model.CLASSES,(tuple,list))

    img_files=os.listdir(test_data_root)
    # img_files=os.listdir(kitti_root)

    for idx,img in enumerate(img_files):
        # img_path=os.path.join(kitti_root,img)
        img_path=os.path.join(test_data_root,img)
        img_name=img
        t1=time.time()
        result = inference_detector(model, img_path)
        t2=time.time()
        print('time is %s'%(t2-t1))
        # savename = savedir + '%s'%img
        img=mmcv.imread(img_path)
        img=img.copy()
        if isinstance(result,tuple):
          bbox_result,segm_result=result
        else:
          bbox_result,segm_result=result,None
        bboxes=np.vstack(bbox_result)
        # draw segmentation masks
        if segm_result is not None:
          segms = mmcv.concat_list(segm_result)
          inds = np.where(bboxes[:, -1] > score_thr)[0]
          for i in inds:
              color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
              mask = maskUtils.decode(segms[i]).astype(np.bool)
              img[mask] = img[mask] * 0.5 + color_mask * 0.5
        labels = [
          np.full(bbox.shape[0], i, dtype=np.int32)
          for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        if score_thr>0:
          assert bboxes.shape[1]==5

          scores=bboxes[:,-1]
          inds=scores>score_thr

          bboxes=bboxes[inds,:]
          labels=labels[inds]
        retina_score_t=0.2
        retina_score_t_7=0.05
        faster_score=0.25
        faster_score_7=0.1
        f=open('./521/add_context/{0}.json'.format(img_name[:-4]),'w')
        # f=open('./res_submit_retina/{0}.json'.format(img_name[:-4]),'w')
        img_bbox=[]

        for bbox,label in zip(bboxes,labels):
            # if bbox[-1]<retina_score_t and label!=7:
            #     continue
            # if bbox[-1]<0.25 :#forretina
            #     continue


            # if bbox[-1]<0.4 and  label in [0,2]:
            #     continue
            # if bbox[-1]<retina_score_t_7:
            #     continue
            write_box={}
            bbox_int = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),round(bbox[4],1)]
            # class_all=list(model.CLASSES)
            label_text = CATEGORIES[
                label]
            write_box[label_text]=bbox_int
            img_bbox.append(write_box)
        json.dump(img_bbox, f, cls=NpEncoder)
        f.close()
        # imwrite(img, savename)

        print('img_{} write is completed!'.format(img_name))


from enum import Enum
from mmcv.utils import is_str
class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)
def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert channel >= 0 and channel <= 255
        return color
    elif isinstance(color, int):
        assert color >= 0 and color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError('Invalid type for color: {}'.format(type(color)))

if __name__ == '__main__':
    main()