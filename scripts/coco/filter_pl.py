import numpy as np
import json
import pickle
import argparse
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(
            description='filtering results')
    parser.add_argument('rname')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    rname = args.rname
    
    gt = mmcv.load('data/coco/annotations/instances_train2014.json')
    class_num = [0] * 91
    for i in range(80):
        class_num[gt['categories'][i]['id']] = i
    
    gtval = mmcv.load('data/coco/annotations/instances_valminusminival2014.json')
    gt_num = [0] * 80
    for i in gtval['annotations']:
        gt_num[class_num[i['category_id']]] += 1
    gt_num = np.array(gt_num)
    gt_num = gt_num / len(gtval['images'])
    
    print(len(gtval['annotations']), len(gtval['images']))
    
    aaa = mmcv.load(rname.split('.')[0] + '.bbox.json')
    
    bc = [[] for i in range(80)]
    for bbox in aaa:
        bc[class_num[bbox['category_id']]].append(bbox)
    
    
    thresh_list = np.linspace(0.1, 1.0, 91)
    thresh = np.zeros((80))
    
    for nc in range(80):
        r = np.zeros((91))
        for t in range(len(thresh_list)):
            ind = [i for i in bc[nc] if i['score'] > thresh_list[t]]
            r[t] = len(ind)/len(gt['images'])
    
        ii = np.argmin(np.abs(r-gt_num[nc]))
        thresh[nc] = thresh_list[ii]
        print(nc, thresh[nc])
    
    
    cci = [i['id'] for i in gt['categories']]
    c = []
    
    
    for i in range(len(aaa)):
        if aaa[i]['score'] > thresh[cci.index(aaa[i]['category_id'])]:
            c.append(aaa[i])
    
    print(len(c))
    
    json.dump(c, open(rname.split('.')[0] + '_t.bbox.json','w'))
    
    a = pickle.load(open(rname, 'rb'))
    c = a.copy()
    
    for i in range(0, len(a)):
        for nc in range(80):
            ind = np.where(a[i][nc][:,-1]>thresh[nc])[0]
            c[i][nc] = a[i][nc][ind]
    
    print(sum([np.vstack(i).shape[0] for i in c]))
    import pickle
    pickle.dump(c, open(rname.split('.')[0] + '_t.pkl','wb'))
    
