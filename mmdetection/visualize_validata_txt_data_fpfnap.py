from __future__ import division
from collections import defaultdict
#这里的defaultdict(function_factory)构建的是一个类似dictionary的对象，
# 其中keys的值，自行确定赋值，但是values的类型，是function_factory的类实例，而且具有默认值。
# 比如default(int)则创建一个类似dictionary对象，里面任何的values都是int的实例，
# 而且就算是一个不存在的key, d[key] 也有一个默认值，这个默认值是int()的默认值0.
import itertools
import numpy as np
import cv2
import six
import json
import os
import argparse


#可视化fn和fp
CLASS={'car'}
BASE_DIR='/media/wrc/0EB90E450EB90E45/data/car_only/occlude/coco/val2017'



def parse_args():
    parser=argparse.ArgumentParser(description='yuyijie eval')
    # parser.add_argument('--pred',help='predict results dir',default='./tools/res/')occ_repul_a7_b3 bi_balance0.5_occ
    parser.add_argument('--pred',help='predict results dir',default='/media/wrc/0EB90C850EB90C85/kitti/mmdetection/biye_pkl/occ_repul_a7_b3.pkl.bbox.json')
    parser.add_argument('--gt',help='ground truth results json',default='/media/wrc/0EB90E450EB90E45/data/car_only/occlude/coco/annotations/instances_val2017.json')
    parser.add_argument('--metric',default='voc')
    parser.add_argument('--th',help='iou_threshold',type=float,default=0.5)
    parser.add_argument('--img_dir',help='the images used to predict',default='/media/wrc/0EB90E450EB90E45/data/car_only/occlude/coco/val2017/')
    parser.add_argument('--save_dir',help='save the fp&fn target image fragment',default='./save_eval/')
    parser.add_argument('--save',help='save the fp&fn target image fragment or not',type=bool,default=True)

    args=parser.parse_args()
    return args




def bbox_iou(bbox_a,bbox_b):
    #防止数据错误
    if bbox_a.shape[1]!=4 or bbox_b.shape[1]!=4:
        raise IndexError

    tl=np.maximum(bbox_a[:,None,:2],bbox_b[:,:2])
    #tl为左上角坐标最大值,为了利用numpy的广播机制,
    # ba[:,None,:2]会得到一个(N,1,2)shape的数组,bb[:,:2]会得到一个(K,2)shape的数组
    #由np的广播性质 两个数组shape都会编成(N,K,2) 也就是对a的每个box都会分别和b的每个box秋左上角
    #坐标最大值
    br=np.minimum(bbox_a[:,None,2:],bbox_b[:,2:])
    #返回的数组的形状将会是(N,K,2)
    area_i=np.prod(br-tl,axis=2)*(tl<br).all(axis=2)
    #首先prod是返回给定轴上数组元素的乘积 (N,K,2)将编成(N,K) 将会少调最后一个轴
    #当tl<br的时候 返回(y1max-y1min)*(xmax-xmin) 即bboxa和bboxb相交的区域
    area_a=np.prod(bbox_a[:,2:]-bbox_a[:,:2],axis=1)#(100,)
    area_b=np.prod(bbox_b[:,2:]-bbox_b[:,:2],axis=1)#(200,)
    return area_i/(area_a[:,None]+area_b-area_i)
    #计算iou 将会是(N,K)纬度的输出,如果所有tl都大于br的话



def eval_detection_voc(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_occ,img_names,gt_difficults=None,
                       iou_thresh=0.5,use_07_metric=False,save=None):
    #根据PASCAL VOC evaluation
    #所有参数都是list
    # test_num张图片（图片数据来自测试数据testdata）的预测框，标签，分数，和真实的框，标签和分数。所有参数都是list
    # len（list）=opt.test_num(default=10000)
    # pred_boxes: [(A, 4), (B, 4), (C, 4)....共test_num个]
    # 输入源gt_数据
    # 经过train.predict函数预测出的结果框
    # pred_labels[(A,), (B,), (C,)...共test_num个]
    # pred_scores同pred_labels
    # A, B, C, D是由nms决定的个数，即预测的框个数，不确定。
    # gt_bboxes：[(a, 4), (b, 4)....共test_num个]
    # a b...是每张图片标注真实框的个数
    # gt_labels与gt_difficults同理
    prec,rec=calc_detection_voc_prec_rec(#计算每个label类别的准确率和召回率
        pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_occ,img_names,gt_difficults,save=save,iou_thresh=iou_thresh
    )
    ap=calc_detection_voc_ap(prec,rec,use_07_metric=use_07_metric)#根据prec和rec计算map和ap
    print({'ap':ap,'map':np.nanmean(ap)})
    return {'ap':ap,'map':np.nanmean(ap)}







def calc_detection_voc_prec_rec(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_occ,img_names,gt_difficults=None,save=None,iou_thresh=0.5):
    pred_bboxes=iter(pred_bboxes)#生成迭代器
    pred_labels=iter(pred_labels)#生成迭代器
    pred_scores=iter(pred_scores)#生成迭代器
    gt_bboxes=iter(gt_bboxes)#生成迭代器
    gt_labels=iter(gt_labels)#生成迭代器
    gt_occs=iter(gt_occ)
    img_name=iter(img_names)
    if gt_difficults is None:
        gt_difficults=itertools.repeat(None)#itertools.repeat生成一个重复的迭代器 None是每次迭代获得的数值
    else:
        gt_difficults=iter(gt_difficults)
    n_pos=defaultdict(int)#defaultdict当key不存在时，dict[key]=default(int)=0 default(list)=[]
    score=defaultdict(list)
    match=defaultdict(list)

    #這個最大的for循環是在遍歷所有圖片

    FP_occ1_num=0
    FN_occ1_num=0
    FP_occ2_num=0
    FN_occ2_num=0
    FP_occ3_num=0
    FN_occ3_num=0
    for pred_bbox,pred_label,pred_score,gt_bbox,gt_label,gt_difficult,gt_occ_local,img, in six.moves.zip(pred_bboxes,pred_labels,
                                                                pred_scores,gt_bboxes,gt_labels,gt_difficults,gt_occs,img_name):

        if gt_difficults is None:
            gt_difficults=np.zeros(gt_bbox.shape[0],dtype=bool)#全部设置为非difficult
        #遍历一边图片中，所有出现的label

        for l in np.unique(np.concatenate((pred_label,gt_label)).astype(int)):
            # FP_target=list()
            # FN_target=list()

            FN_target = defaultdict(list)
            FP_target = defaultdict(list)
            FN_target_occ1 = defaultdict(list)
            FN_target_occ2 = defaultdict(list)
            FP_target_occ1 = defaultdict(list)
            FP_target_occ2 = defaultdict(list)
            #拼接后返回无重复的从小到大排序的一维numpy 如[2,3,4,5,6]
            # 并遍历这个一维数组，即遍历这张图片出现过的标签数字(gt_label+pred_label)
            #>>> np.unique([1, 1, 2, 2, 3, 3])
            # array([1, 2, 3])
            # >>> a = np.array([[1, 1], [2, 3]])
            # >>> np.unique(a)
            # array([1, 2, 3])
            pred_mask_l=pred_label==l#//广播pred_mask_l=[eg. T,F,T,T,F,F,F,T..] 所有预测label中等于L的为T 否则F
            pred_bbox_l=pred_bbox[pred_mask_l]#选出label=L的所有pre_box
            pred_score_l=pred_score[pred_mask_l]#label=L 对应所有pre_score

            #sort by score
            order=pred_score_l.argsort()[::-1]#获得score降序排序索引
            pred_bbox_l=pred_bbox_l[order]
            pred_score_l=pred_score_l[order]

            gt_mask_l=gt_label==l#同理
            gt_bbox_l=gt_bbox[gt_mask_l]
            gt_occ_l=gt_occ_local[gt_mask_l]
            gt_occ_l_1=np.where(gt_occ_l==1)
            gt_occ_l_2=np.where(gt_occ_l==2)
            # gt_difficult_l=gt_difficult[gt_mask_l]
            n_pos[l]+=len(gt_bbox_l)

            # n_pos[l]+=np.logical_not(gt_difficult_l).sum()#对T，F取反求和，统计出difficult=0的个数
            score[l].extend(pred_score_l)#score={l:predscore_l,...} extend是针对defaultdict中是list的情况

            if len(pred_bbox_l)==0:#没有预测的label=L的box，即真实label有L，我们全没有预测到
                FN_target[l].extend(gt_bbox_l.tolist())
                if save is not None and len(FN_target)>0:
                    save_target(FN_target,FP_target,img[0],l,save)
                continue#跳过这张图片 此时没有对match字典操作，之前score[l].extend操作也为空 保持了match和score的形状一致

            if len(pred_bbox_l)==0 and gt_occ_l_1.size>0:
                FN_target_occ1[l].extend(gt_bbox_l[gt_occ_l_1].tolist())
                if save is not None and len( FN_target_occ1) > 0:
                    save_target( FN_target_occ1, FP_target, img[0], l, save)
                continue  # 跳过这张图片 此时没有对match字典操作，之前score[l].extend操作也为空 保持了match和score的形状一致
            if len(pred_bbox_l)==0 and gt_occ_l_2.size>0:
                FN_target_occ2[l].extend(gt_bbox_l[gt_occ_l_2].tolist())
                if save is not None and len( FN_target_occ2) > 0:
                    save_target( FN_target_occ2, FP_target, img[0], l, save)
                continue  # 跳过这张图片 此时没有对match字典操作，之前score[l].extend操作也为空 保持了match和score的形状一致


            if len(gt_bbox_l)==0:#没有真实的label=L的情况 即预测中有L 真实中没有 全都预测错了
                match[l].extend((0,)*pred_bbox_l.shape[0])#match{L:[0,0,0...n_pred_box个0]}
                FP_target[l].extend(pred_bbox_l.tolist())
                if save is not None and len(FP_target)>0:
                    save_target(FN_target,FP_target,img[0],l,save)
                continue

            # VOC evaluation follows integer typed bounding boxes.
            # 作者给的注释是follows integer typed bounding boxes
            # 但是只改变了ymax, xmax的值，重要的是这样做并不能转化为整数
            # pred_bbox和gt_bbox只
            # 参与了IOU计算且后面没有参与其他计算
            pred_bbox_l=pred_bbox_l.copy()
            pred_bbox_l[:,2:]+=1#ymax,xmax+=1
            gt_bbox_l=gt_bbox_l.copy()
            gt_bbox_l[:,2:]+=1
            min_sides=np.minimum(pred_bbox_l[:,2]-pred_bbox_l[:,0],pred_bbox_l[:,3]-pred_bbox_l[:,1])
            iou=bbox_iou(pred_bbox_l,gt_bbox_l)#计算两个box的iou
            gt_index=iou.argmax(axis=1)#有len(pred_bbox_l)个索引，第i个索引值n表示gt_box[n]与pred_box[i]iou最大
            #比如 4个pred_box 3 个gtbox
            #     gt0 gt1 gt2    最大对应用*表示   这里的gt_index输出将是[1,0,0,2] 第0个索引值1 表示gt1和pred_box[0]iou最大
            #  A       *
            #  B   *
            #  C   *
            #  D           *
            #要注意 这里的gt_index是会重复的 因为同一个gt会与很多pred_box拥有最大iou
            #    #计算iou 将会是(N,K)纬度的输出,如果所有tl都大于br的话
            dict_dup_fp=defaultdict(list)
            for pre_idx,gt_idx in enumerate(gt_index.tolist()):
                dict_dup_fp[gt_idx].extend([pre_idx])#把fp的bbox收集起來
            for key,val in dict_dup_fp.items():
                if len(val)>=2:
                    scores_pred_for_one_gt=pred_score_l[val]
                    scores_sort_index_without_max=scores_pred_for_one_gt.argsort()[:-1]
                    FP_target[l].extend(pred_bbox_l[scores_sort_index_without_max].tolist())
                    if gt_occ_l[key]==1:
                        FP_target_occ1[l].extend(pred_bbox_l[scores_sort_index_without_max].tolist())
                    elif gt_occ_l[key]==2:
                        FP_target_occ2[l].extend(pred_bbox_l[scores_sort_index_without_max].tolist())

            gt_detected=np.unique(gt_index)
            gt_missed_FN=list()
            for i in range(len(gt_bbox_l)):
                if i not in gt_detected:
                    gt_missed_FN.append(i)#把FN的bbox收集起來
                    FN_target[l].extend([gt_bbox_l[i].tolist()])
                    if gt_occ_l[i]==1:
                        FN_target_occ1[l].extend([gt_bbox_l[i].tolist()])
                    elif gt_occ_l[i]==2:
                        FN_target_occ2[l].extend([gt_bbox_l[i].tolist()])
            if save is not None and (len(FP_target)>0 or (len(FN_target)>0)):
                save_target(FN_target,FP_target,img[0],l,save)
                if len(FN_target_occ1)>0 or (len(FP_target_occ1)>0):
                    save_target(FN_target_occ1,FP_target_occ1,img[0],l,save+'/occ1')
                if len(FN_target_occ2)>0 or (len(FP_target_occ2)>0):
                    save_target(FN_target_occ2,FP_target_occ2,img[0],l,save+'/occ2')

                save_target2(pred_bbox_l,img[0],l,save+'/ori')
            FP_occ1_num+=len(FP_target_occ1)
            FP_occ2_num+=len(FP_target_occ2)
            FN_occ1_num+=len(FN_target_occ1)
            FN_occ2_num+=len(FN_target_occ2)

            #加入gt_box最短边的限制 可能有多个gtbox
            # iou_thresh_new=[]
            # for min_side in min_sides:#跟gt同顺序
            #     if min_side>=420:
            #         iou_thresh_new.append(0.8)
            #     elif 120<=min_side<420:
            #         iou_thresh_new.append(min_side/1500+0.52)
            #     elif 40<=min_side<120:
            #         iou_thresh_new.append(min_side/200)
            #     else:
            #         iou_thresh_new.append(0.2)
            # iou_thresh_new=np.array(iou_thresh_new)
            # gt_index[iou.max(axis=1)<iou_thresh_new]=-1
            gt_index[iou.max(axis=1)<iou_thresh]=-1
            #这里则是在上面的基础上把小于阈值的gt_index设置为-1
            #将gt_box与pred_box iou<thresh的索引值置为-1
             # 即针对每个pred_bbox，与每个gt_bbox IOU的最大值 如果最大值小于阀值则置为-1
             # 即我们预测的这个box效果并不理想 后续会将index=-1的 matchlabel=0
            del iou
            selec=np.zeros(gt_bbox_l.shape[0],dtype=bool)
            for gt_idx in gt_index:#遍历gt_index索引值
                if gt_idx>=0:#即iou满足条件的bbox

                    if not selec[gt_idx]:#没有被选国，select[idx]=0的时候
                        match[l].append(1)
                    else:#对应的gt_box已经被选国一次，即已经和前面的某个pre_box iou最大
                        match[l].append(0)
                    selec[gt_idx]=True#一个gt被选过之后，则要设置为True
                else:#不满足iou>thresh
                    match[l].append(0)#追加0 很重复匹配的结果一样
    print('occ_fp1:{},occ_fn1:{},occ_fp2:{},occ_fn2:{}'.format(FP_occ1_num,FN_occ1_num,FP_occ2_num,FN_occ2_num))
    #我们注意到上面为每个pred_box都打了label 0, 1, -1
    #len(match[l]) = len(score[l]) = len(pred_bbox_l)
    for iter_ in ( # 上面的 six.moves.zip遍历会在某一iter遍历到头后停止，由于pred_bboxes等是全局iter对象，
    #我们此时继续调用next取下一数据，如果有任一数据不为None，那么说明他们的len是不相等的 有悖常理，数据错误
    pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels):
        if next(iter_,None) is not None:#next(iter_,None)表示调用next 如果已经遍历到了头 不抛出异常而是返回None
            raise ValueError("Length of input iterables need to be same")

    n_fg_class=max(n_pos.keys())#有n_fg_class个类
    prec=[None]*n_fg_class
    rec=[None]*n_fg_class
    for l in n_pos.keys():#遍历所有label
        score_l=np.array(score[l])
        match_l=np.array(match[l],dtype=np.int8)
        order=score_l.argsort()[::-1]
        match_l=match_l[order]#对应match按照score由大到小排序

        tp=np.cumsum(match_l==1)#统计累计match_1=1的个数
        # 比如 match_l =[1,1,1,0,1,1] np.cumsum(match==1)=[1,2,3,3,4,5]
        # tp=[1,2,3,3,4,5] fp=[0,0,0,1,1,1] 长度是所有predbox的总数 prec[l]=[1,1,1,0.75,0.8,0.833]
        fp=np.cumsum(match_l==0)
        prec[l-1]=tp/(tp+fp)
        if n_pos[l]>0:#如果n_pos[l]=0 那么rec[l]=None
            rec[l-1]=tp/n_pos[l]#这里干嘛不用所有的gt[l]的总数
    return prec,rec

def calc_detection_voc_ap(prec,rec,use_07_metric=True):
    n_fg_class=len(prec)
    ap=np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):#遍历每个label
        if prec[l] is None or rec[l] is None:#如果为NOne 则ap设置为np.nan
            ap[l]=np.nan
            continue
        if use_07_metric:
            # 11 point metric
            ap[l]=0
            for t in np.arange(0.,1.1,0.1):#t=0 0.1 0.2...1.0
                if np.sum(rec[l]>=t)==0:#这个标签召回率没有大于阈值的
                    p=0
                else:
                    p=np.max(np.nan_to_num(prec[l])[rec[l]>=t])#p=(rec>=t时，对应index：prec中的最大值) np.nan_to_num是为了
                    #让None=0一边计算
                ap[l]+=p/11
        else:
            mpre=np.concatenate(([0],np.nan_to_num(prec[l]),[0]))#头尾添加0
            mrec=np.concatenate(([0],rec[l],[1]))#头添加0 尾添加1
            mpre=np.maximum.accumulate(mpre[::-1])[::-1]#获得从小到大的累计最大值


            #>>> np.add.accumulate([2, 3, 5])
            # array([ 2,  5, 10])
            # >>> np.multiply.accumulate([2, 3, 5])
            # array([ 2,  6, 30])
            # 我们知道
            # 我们是按score由高到低排序的
            # 而且我们给box打了label
            # 0, 1, -1
            # score高时1的概率会大，所以pre是累计降序的
            # 而rec是累积升序的，那么此时将pre倒序再maxuim.ac
            # 获得累积最大值，再倒序后
            # 从小到大排序的累积最大值
            i=np.where(mrec[1:]!=mrec[:-1])[0]#差位比较，看哪里改变了recall的值，记录index（x轴）
            #and sum (\Delta recall)*prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])#差值*mpre_max的值，（x轴之差*ymax）
    return ap


def save_target(FNtarget_bbox,FPtarget_bbox,img_name,category,save_path):
    img_dir=os.path.join(BASE_DIR,img_name)
    img=cv2.imread(img_dir)
    # if img_name=='000674.png':
    #     print(FNtarget_bbox)
    for l,fn_t in FNtarget_bbox.items():
        for box in fn_t:
            try:
                cv2.rectangle(img,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(0,0,255),2)
            except:
                raise TypeError('box is {0} ,FNtarget_BOX is {1}'.format(box,fn_t))
            font=cv2.FONT_HERSHEY_DUPLEX
            # text=str(l)+'fn'
            # cv2.putText(img,text,(int(box[1]),int(box[0]-5)),font,2,(0,255,0),1)
    for l,fp_t in FPtarget_bbox.items():
        for box in fp_t:
            cv2.rectangle(img,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(0,255,255),2)
            font=cv2.FONT_HERSHEY_DUPLEX
            # text=str(l)+'fp'
            # cv2.putText(img,text,(int(box[1]),int(box[0]-5)),font,2,(0,255,255),2)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path=os.path.join(save_path,img_name)
    cv2.imwrite(save_path,img)


def save_target2(bbox,img_name,category,save_path):
    img_dir=os.path.join(BASE_DIR,img_name)
    img=cv2.imread(img_dir)
    # if img_name=='000674.png':
    #     print(FNtarget_bbox)
    for box in bbox:

        cv2.rectangle(img,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(0,255,0),2)

        font=cv2.FONT_HERSHEY_DUPLEX
        # text='car'
        # cv2.putText(img,text,(int(box[1]),int(box[0]-5)),font,2,(255,0,0),2)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path1=os.path.join(save_path,img_name)
    cv2.imwrite(save_path1,img)






if __name__=='__main__':
    from collections import OrderedDict
    from collections import defaultdict
    args=parse_args()
    gt_dict=json.load(open(args.gt,'r'))
    gt_box=defaultdict(list)
    gt_labels=defaultdict(list)
    gt_occ=defaultdict(list)
    gt_imgs=defaultdict(str)
    score_thresh=0.5
    for key,value in gt_dict.items():
        if key=='images':
            for img_dic in value:#遍歷images list中的每個字典圖片
                gt_imgs[img_dic['id']]=img_dic['file_name']
        if key=='annotations':
            for anno_lst in value:#遍歷anno list中每個字典標籤
                # gt_box[gt_imgs[anno_lst['image_id']]].append(anno_lst['bbox'])
                ymin=anno_lst['bbox'][1]+1
                xmin=anno_lst['bbox'][0]+1
                ymax=anno_lst['bbox'][1]+1+anno_lst['bbox'][3]
                xmax=anno_lst['bbox'][0]+1+anno_lst['bbox'][2]
                gt_occ[gt_imgs[anno_lst['image_id']]].extend([int(anno_lst['occluded'])])
                gt_box[gt_imgs[anno_lst['image_id']]].append([ymin,xmin,ymax,xmax])
                gt_labels[gt_imgs[anno_lst['image_id']]].extend([anno_lst['category_id']])


    pred_box=defaultdict(list)
    pred_label=defaultdict(list)
    pred_score=defaultdict(list)
    pred_imgs=defaultdict(str)

    pred_dict=json.load(open(args.pred,'r'))

    # files=os.listdir(pred_dir)
    for box_dict in pred_dict:
        if box_dict['image_id'] not in pred_imgs :
            pred_imgs[box_dict['image_id']]=gt_imgs[box_dict['image_id']]
        if box_dict['score']<score_thresh:
            continue
        ymin=box_dict['bbox'][1]+1
        xmin=box_dict['bbox'][0]+1
        ymax=box_dict['bbox'][1]+1+box_dict['bbox'][3]
        xmax=box_dict['bbox'][0]+1+box_dict['bbox'][2]
        pred_box[pred_imgs[box_dict['image_id']]].append([ymin,xmin,ymax,xmax])
        pred_label[pred_imgs[box_dict['image_id']]].extend([box_dict['category_id']])
        pred_score[pred_imgs[box_dict['image_id']]].extend([box_dict['score']])

    img_name_=[]
    pred_bbox_=[]
    pred_labels_=[]
    pred_scores_=[]
    gt_bbox_=[]
    gt_label_=[]
    gt_occ_=[]
    for key_gt,value_gt in gt_imgs.items():
        for key_pred,value_pred in pred_imgs.items():
            if value_gt==value_pred :
                img_name_.append([value_gt])
                pred_bbox_.append(np.array(pred_box[value_pred]))
                gt_bbox_.append(np.array(gt_box[value_gt]))
                pred_labels_.append(np.array(pred_label[value_pred]))
                gt_label_.append(np.array(gt_labels[value_gt]))
                gt_occ_.append(np.array(gt_occ[value_gt]))
                pred_scores_.append(np.array(pred_score[value_pred]))
    print('loading the results!')

    eval_detection_voc(pred_bbox_,pred_labels_,pred_scores_,gt_bbox_,gt_label_,gt_occ_,img_name_,save='./save_eval_rep')









