import sys
import numpy as np
import pickle
from copy import deepcopy
from ensemble_boxes import weighted_boxes_fusion
import warnings 
import argparse
warnings.filterwarnings("ignore") 

def parse_args():
    parser = argparse.ArgumentParser(
            description='ensembling')
    parser.add_argument('r1')
    parser.add_argument('r2')
    parser.add_argument('r3')
    parser.add_argument('r_ensemble')
    args = parser.parse_args()
    return args

def transform(vd):
    bboxes = []
    scores = []
    labels = []
    for i in range(len(vd)):
        bbox, score, label = [], [], []
        for j in range(len(vd[i])):
            if vd[i][j].shape[0] > 0:
                for k in range(vd[i][j].shape[0]):
                    bbox.append(vd[i][j][k][0:4])
                    score.append(vd[i][j][k][-1])
                    label.append(j)
        for k in range(len(bbox)):
            bb = bbox[k]
            #bb = np.array([bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]])
            bb = bb/5000 #max(gt[i]['height'], gt[i]['width'])
            bb = bb.tolist()
            bbox[k] = bb
        bboxes.append(bbox)
        scores.append(score)
        labels.append(label)
    return bboxes, scores, labels

def itransform(bboxes, scores, labels):
    vdd =[np.zeros((0,5))] * 80 ###
    for k in range(labels.shape[0]):
        bb = np.zeros((1,5))
        bb[0][0] = bboxes[k][0] * 5000
        bb[0][1] = bboxes[k][1] * 5000
        bb[0][2] = (bboxes[k][2]) * 5000
        bb[0][3] = (bboxes[k][3]) * 5000
        bb[0][4] = scores[k]
        #print(labels[k])
        vdd[int(labels[k])] = np.concatenate([vdd[int(labels[k])], bb])
    return vdd

if __name__ == '__main__':
    args = parse_args()
    r1 = args.r1
    r2 = args.r2
    r3 = args.r3
    r_ensemble = args.r_ensemble

    #weights = [0.9, 1.0, 1.4]
    weights = [0.7, 1.0, 1.7]

    t = 0.6
    ing = 0.0001
    conf = 'avg'

    a = pickle.load(open(r1, 'rb'))
    b = pickle.load(open(r2, 'rb'))
    c = pickle.load(open(r3, 'rb'))

    boxes_list_a, scores_list_a, labels_list_a = transform(a)
    boxes_list_b, scores_list_b, labels_list_b = transform(b)
    boxes_list_c, scores_list_c, labels_list_c = transform(c)
           
    
    d = []
    for i in range(len(a)):
        boxes, scores, labels =  weighted_boxes_fusion([boxes_list_a[i], boxes_list_b[i], boxes_list_c[i]], [scores_list_a[i], scores_list_b[i], scores_list_c[i]], [labels_list_a[i], labels_list_b[i], labels_list_c[i]], weights=weights, iou_thr=t, skip_box_thr=ing, conf_type=conf)
        vdd = itransform(boxes, scores, labels)
        d.append(vdd)
    
    pickle.dump(d, open(r_ensemble,'wb'))


