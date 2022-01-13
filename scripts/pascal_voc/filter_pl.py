import numpy as np
import pickle
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
            description='thresholding')
    parser.add_argument('resultname')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    fname = args.resultname

    gt = pickle.load(open('labels/voc07_trainval.pkl','rb'))
    gtn = np.zeros((20))
    for i in gt:
        for j in i['ann']['labels']:
            gtn[j] += 1
    rgt = gtn/len(gt)
    
    thresh_list = np.linspace(0.1, 1.0, 91)
    
    a = pickle.load(open(fname,'rb'))
    c = a.copy()
    l = len(a)
    
    thresh = np.zeros((20))
    
    for nc in range(20):
        r = np.zeros((91))
        for t in range(len(thresh_list)):
            for i in range(0, l):
                ind = np.where(a[i][nc][:,-1] > thresh_list[t])[0]
                r[t] += len(ind)
        r = r/len(a)
        ii = np.argmin(np.abs(r-rgt[nc]))
        thresh[nc] = thresh_list[ii]
        print(nc, thresh[nc])
    
    for i in range(0, l):
        for nc in range(20):
            ind = np.where(a[i][nc][:,-1]>thresh[nc])[0]
            c[i][nc] = a[i][nc][ind]
    
    pickle.dump(c, open(fname.split('.')[0] + '_t.pkl','wb'))

