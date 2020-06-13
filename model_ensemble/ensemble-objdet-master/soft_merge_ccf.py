import os
import numpy as np

# soft emsemble to keep more detection results
def GeneralEnsemble(dets, iou_thresh=0.5, weights=None,sigma=0.5,Nt=0.2,threshold=0.001):
    assert (type(iou_thresh) == float)

    ndets = len(dets)

    if weights is None:
        w = 1 / float(ndets)
        weights = [w] * ndets
    else:
        assert (len(weights) == ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()


    for idet in range(0, ndets):
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
                    obox_new=obox
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
                    found.append((bestbox, w))
                    used.append(bestbox)

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                new_box[5] = round(new_box[5], 2)
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                iou = computeIOU(box, found[0][0])
                weight = np.exp(-(iou * iou) / sigma)
                found[0][0][5] = found[0][0][5] * weight
                if found[0][0][5]>Nt and found[0][0][5]<threshold:
                    out.append(box)
                    out.append(found[0])

                else:
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
                        xc += w * b[0]
                        yc += w * b[1]
                        bw += w * b[2]
                        bh += w * b[3]
                        conf += w * b[5]

                    xc /= wsum
                    yc /= wsum
                    bw /= wsum
                    bh /= wsum

                    xc = int(xc)
                    yc = int(yc)
                    bw = int(bw)
                    bh = int(bh)
                    conf = round(conf, 2)
                    # conf=format(conf,'.2f')

                    new_box = [xc, yc, bw, bh, box[4], conf]
                    out.append(new_box)
    return out


def getCoords(box):
    x1 = box[0]
    x2 = box[2]
    y1 = box[1]
    y2 = box[3]
    return x1, x2, y1, y2


def computeIOU(box1, box2):
    # x11, x12, y11, y12 = getCoords(box1)
    # x21, x22, y21, y22 = getCoords(box2)
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area+0.1)
    return iou


if __name__ == "__main__":
    images_path=[]
    IMAGES_PATH='D:/postgraduateworking/dataset/KITTI/testing/image_2/'
    #IMAGES_PATH ='D:/postgraduateworking/dataset/ccftest/VOC2007_test用/JPEGImages/'
    model1_path='./cas/'
    model2_path='./retinanet_result/'
    if os.path.isdir(IMAGES_PATH):
        for image_file in os.listdir(IMAGES_PATH):
            images_path+=[IMAGES_PATH+image_file]

    for image_path in images_path:
        _,file=os.path.split(image_path)
        filename,_=os.path.splitext(file)
        det = []
        det1 = []
        det2 = []
        fileObject = open('soft_result_merge/%s.txt' % filename, 'w')
        for result_file1 in os.listdir(model1_path):
            result_name1,_=os.path.splitext(result_file1)
            if result_name1==filename:
                #det1=[]
                with open('cas/%s.txt'%result_name1, 'r') as f:
                    boxes1 = f.read().splitlines()
                    for img_label in boxes1:
                        box=img_label.split('\t')
                        if box[0]=='car':
                            box[0]=str(0)
                        elif box[0]=='pedestrian':
                            box[0]=str(1)
                        elif box[0]=='cyclist':
                            box[0]=str(2)
                        box_list1=[int(box[4]),int(box[5]),int(box[6]),int(box[7]),int(box[0]),float(box[-2])]
                        det1.append(box_list1)


        for result_file2 in os.listdir(model2_path):
            result_name2,_=os.path.splitext(result_file2)
            if result_name2==filename:
                #det2=[]
                with open('retinanet_result/%s.txt'%result_name2, 'r') as f:
                    boxes2 = f.read().splitlines()
                    for img_label in boxes2:
                        box=img_label.split('\t')
                        if box[0]=='Car':
                            box[0]=str(0)
                        elif box[0]=='Pedestrian':
                            box[0]=str(1)
                        elif box[0]=='Cyclist':
                            box[0]=str(2)
                        box_list2=[int(box[4]),int(box[5]),int(box[6]),int(box[7]),int(box[0]),float(box[-2])]
                        det2.append(box_list2)

        det.append(det1)
        det.append(det2)

        print(det)

        ens = GeneralEnsemble(det, weights=[0.85, 0.6])
        label={0:'Car',1:'Pedestrian',2:'Cyclist'}
        for ens_box in ens:
            fileObject.write(label[ens_box[-2]])
            fileObject.write('\t')
            fileObject.write('-1')
            fileObject.write('\t')
            fileObject.write('-1')
            fileObject.write('\t')
            fileObject.write('0.0')
            fileObject.write('\t')
            fileObject.write(str(ens_box[0]))
            fileObject.write('\t')
            fileObject.write(str(ens_box[1]))
            fileObject.write('\t')
            fileObject.write(str(ens_box[2]))
            fileObject.write('\t')
            fileObject.write(str(ens_box[3]))
            fileObject.write('\t')
            fileObject.write('0.0')
            fileObject.write('\t')
            fileObject.write('0.0')
            fileObject.write('\t')
            fileObject.write('0.0')
            fileObject.write('\t')
            fileObject.write('0.0')
            fileObject.write('\t')
            fileObject.write('0.0')
            fileObject.write('\t')
            fileObject.write('0.0')
            fileObject.write('\t')
            fileObject.write('0.0')
            fileObject.write('\t')
            fileObject.write(str(ens_box[-1]))
            fileObject.write('\t')
            fileObject.write('\n')
        fileObject.close()
        print(ens)









