#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Navinfo
# @File    : convert.py
""""
Run convert.y to convert detection and segmentation to submit format
"""
import cv2
import numpy as np
import pandas as pd
import argparse

def rle_encode(img):
    img = img.flatten()
    label_dict = {0:""}
    for id in range(26,35):
        label_dict[id] = ""
    preid = img[0]
    id_num = 1
    begin_idx = 0
    for pixel_idx in range(1,img.shape[0]):
        pixel = img[pixel_idx]
        if preid == pixel:
            id_num += 1
        else:
            label_dict[preid] = label_dict[preid] + "{} {}|".format(begin_idx,id_num)
            begin_idx = pixel_idx
            id_num = 1
        preid = pixel
    return label_dict

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_txt",type=str,default='./train_new.txt',help="the path of result.txt")
    parser.add_argument("--output_path",type=str,default='sample_submission.csv',help="the path of output csv")
    return parser.parse_args()

def main(args):
    df = pd.DataFrame(columns=['image_id','task','label','pixelcount','result'])
    results = open(args.result_txt,'r').readlines()
    pd_idx = 0
    for line in results:
        result_list = line.strip().split(' ')
        img_path,segment_path = result_list[:2]
        detection_results = result_list[2:]
        for detection_result in detection_results:
            result = ",".join(detection_result.split(',')[:4]) + ',' + detection_result.split(',')[-1]
            label = detection_result.split(',')[-2]
            df.loc[pd_idx]= img_path,0,label,None,result
            pd_idx += 1
        print('./seg_label/' + segment_path)
        seg_result = cv2.imread('./seg_label/' + segment_path,0)
        label_dict = rle_encode(seg_result)
        for label in range(26,35):
            if label_dict[label] != '':
                PixelCount = np.where(seg_result==label)[0].shape[0]
                df.loc[pd_idx]= img_path,1,label,PixelCount,label_dict[label].strip('|')
                pd_idx += 1

    df.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    args = init_args()
    main(args)



