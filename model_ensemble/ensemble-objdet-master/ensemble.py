""" 
Ensembling methods for object detection.
"""
import os
import numpy as np
""" 
General Ensemble - find overlapping boxes of the same class and average their positions
while adding their confidences. Can weigh different detectors with different weights.
No real learning here, although the weights and iou_thresh can be optimized.

Input: 
 - dets : List of detections. Each detection is all the output from one detector, and
          should be a list of boxes, where each box should be on the format 
          [box_x, box_y, box_w, box_h, class, confidence] where box_x and box_y 
          are the center coordinates, box_w and box_h are width and height resp.
          The values should be floats, except the class which should be an integer.

 - iou_thresh: Threshold in terms of IOU where two boxes are considered the same, 
               if they also belong to the same class.
               
 - weights: A list of weights, describing how much more some detectors should
            be trusted compared to others. The list should be as long as the
            number of detections. If this is set to None, then all detectors
            will be considered equally reliable. The sum of weights does not
            necessarily have to be 1.

Output:
    A list of boxes, on the same format as the input. Confidences are in range 0-1.
"""
CATEGORIES = {
    0:'Consolidation',1: 'Fibrosis', 2:'Effusion', 3: 'Nodule', 4:'Mass',
    5:'Emphysema', 6:'Calcification', 7:'Atelectasis', 8:'Fracture'
}
CATEGORIES2 = {
    'Consolidation':0, 'Fibrosis':1, 'Effusion':2, 'Nodule':3, 'Mass':4,
    'Emphysema':5, 'Calcification':6, 'Atelectasis':7, 'Fracture':8
}
import json

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

def GeneralEnsemble(dets, iou_thresh = 0.5, weights=None):
    assert(type(iou_thresh) == float)
    
    ndets = len(dets)
    
    if weights is None:
        w = 1/float(ndets)
        weights = [w]*ndets
    else:
        assert(len(weights) == ndets)
        
        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()
    
    for idet in range(0,ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue
                
            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]
                
                if odet == det:
                    continue
                
                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox
                                
                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox,w))
                    used.append(bestbox)
                            
            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)
                
                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0
                
                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w*b[0]
                    yc += w*b[1]
                    bw += w*b[2]
                    bh += w*b[3]
                    conf += w*b[5]
                
                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum    

                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out
    
def getCoords(box):
    x1 = float(box[0]) - float(box[2])/2
    x2 = float(box[0]) + float(box[2])/2
    y1 = float(box[1]) - float(box[3])/2
    y2 = float(box[1]) + float(box[3])/2
    return x1, x2, y1, y2
    
def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)
    
    x_left   = max(x11, x21)
    y_top    = max(y11, y21)
    x_right  = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0    
        
    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)        
    
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou

def Parse_anno(file):
        dets = []
        with open(file) as f:
            ann=json.load(f)
            for dic in ann:
                for k,v in dic.items():
                    w=v[2]-v[0]
                    h=v[3]-v[1]
                    x_c=v[0]+w/2
                    y_c=v[1]+h/2
                    dets.append([x_c,y_c,w,h,CATEGORIES2[k],v[-1]])
        return dets


score1 = 0.53
score2 = 0.71
score3=0.54
if __name__=="__main__":
    # Toy example
    # savedir = './saved_images_submit_ensemble/'

    # model1_path='/home/wrc/Competition/mmdetection-master/res_submit_mix_final'
    model1_path='/home/wrc/Competition/mmdetection-master/521/cas'
    model2_path='/home/wrc/Competition/mmdetection-master/521/mix'
    model3_path='/home/wrc/Competition/mmdetection-master/521/sepc'
    # model3_path='/home/wrc/Competition/mmdetection-master/520/mosaic'
    # model3_path='/home/wrc/Competition/mmdetection-master/520/mosaic_val'
    # model3_path='/home/wrc/Competition/mmdetection-master/520/mosaic_val'
    json_file1=os.listdir(model2_path)
    for file in json_file1:
        dets=[]
        json1=os.path.join(model1_path,file)
        json2=os.path.join(model2_path,file)
        json3=os.path.join(model3_path,file)
        dets.append(Parse_anno(json1))
        dets.append(Parse_anno(json2))
        dets.append(Parse_anno(json3))


        ens = GeneralEnsemble(dets, weights = [score1,score2, score3])
        # ens = GeneralEnsemble(dets, weights = [retina_score, mix_score])
        # ens = GeneralEnsemble(dets, weights = [score2, score3])
        f = open('/home/wrc/Competition/mmdetection-master/521/ensemble_cas/{0}.json'.format(file[:-5]), 'w')
        img_bbox = []
        for bbox in ens:
            write_box={}
            xmin=bbox[0]-bbox[2]/2
            ymin=bbox[1]-bbox[3]/2
            xmax=bbox[0]+bbox[2]/2
            ymax=bbox[1]+bbox[3]/2
            bbox_int = [int(xmin),int(ymin),int(xmax),int(ymax),round(bbox[-1],1)]
            if bbox_int[-1]<0.1:
                continue
            label_text = CATEGORIES[
                bbox[-2]]
            write_box[label_text]=bbox_int
            img_bbox.append(write_box)
        json.dump(img_bbox, f, cls=NpEncoder)
        f.close()
        print(ens)
