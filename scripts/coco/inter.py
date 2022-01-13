import sys
import numpy as np
import pickle
import argparse
from mmcv.ops.nms import nms
from copy import deepcopy
import os

def parse_args():
    parser = argparse.ArgumentParser(
            description='intering')
    parser.add_argument('result1')
    parser.add_argument('result2')
    args = parser.parse_args()
    return args


def iou(bbox1, bbox2):
    bo = [max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])]
    if bo[2]<=bo[0] or bo[3]<=bo[1]:
        return 0.0
    so = (bo[2] - bo[0]) * (bo[3] - bo[1])
    s2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    s1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    return so/(s1+s2-so)

if __name__ == '__main__':
    args = parse_args()
    result1 = args.result1
    result2 = args.result2
    a = pickle.load(open(result1.split('.')[0] + '_t.pkl', 'rb'))
    b = pickle.load(open(result2.split('.')[0] + '_t.pkl', 'rb'))

    s = []

    pb = [np.zeros((0,5)).astype('float32') for i in range(80)]   ####
    c = [deepcopy(pb) for i in range(len(a))]

    for i in range(len(a)):
        for nc in range(80):
            if b[i][nc].shape[0] == 0 and a[i][nc].shape[0] == 0:
                pass
            elif b[i][nc].shape[0] == 0:
                pass
            elif a[i][nc].shape[0] == 0:
                pass
            else:
                kb = b[i][nc].shape[0]
                ka = a[i][nc].shape[0]
                for k1 in range(ka):
                    for k2 in range(kb):
                        if iou(a[i][nc][k1], b[i][nc][k2]) > 0.5:
                            if a[i][nc][k1,-1] < b[i][nc][k2,-1]:
                                bb = b[i][nc][k2]
                            else:
                                bb = a[i][nc][k1]
                            bb = bb[np.newaxis,:]
                            c[i][nc] = np.concatenate([c[i][nc], bb])
    
    for i in range(len(a)):
        for j in range(len(a[i])):
            c[i][j] = nms(c[i][j][:, 0:4].astype('float32'), c[i][j][:,-1].astype('float32'), 0.3)[0]
    
    pickle.dump(c, open(result2.split('.')[0] + '_t.pkl','wb'))
    print('intering done')
    os.system('python scripts/coco/pkl2json.py  ' + result2.split('.')[0] + '_t.pkl')


