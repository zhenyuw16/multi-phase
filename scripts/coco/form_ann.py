import json
from copy import deepcopy
import numpy as np
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description='forming annotation file')
    parser.add_argument('plname')
    parser.add_argument('annname')
    parser.add_argument('ratio')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    fann = args.annname
    fpl = args.plname
    ratio = float(args.ratio)
    
    pls = pickle.load(open(fpl.split('.')[0] + '_t.pkl', 'rb'))
    dett = json.load(open(fpl.split('.')[0] + '_t.bbox.json'))
    
    a = json.load(open('data/coco/annotations/instances_train2014.json'))
    aaa = json.load(open('data/coco/annotations/instances_valminusminival2014.json'))
    
    iiid = [i['id'] for i in aaa['annotations']]
    print(len(a['annotations']))
    
    average_score = np.array([np.mean(np.vstack(i)[:,-1]) for i in pls])
    average_score[np.isnan(average_score)] = -1.
    num = np.sum(average_score==-1)
    
    score_ranking = np.argsort(-average_score)
    filters = score_ranking[0: int((score_ranking.shape[0]-num)*ratio)]

    ind = np.array([a['images'][i]['id'] for i in filters])
    b = [i for i in dett if i['image_id'] in ind]

    j = max(iiid)
    for i in range(len(b)):
        x1, x2, y1, y2 = [b[i]['bbox'][0], b[i]['bbox'][0]+b[i]['bbox'][2], b[i]['bbox'][1], b[i]['bbox'][1]+b[i]['bbox'][3]]
        b[i]['segmentation'] = [[ ]]
        b[i]['area'] = b[i]['bbox'][2] * b[i]['bbox'][3]
        j = j + 1
        b[i]['id'] = j
    
    a['annotations'] = b
    
    im = [i for i in a['images'] if i['id'] in ind]
    a['images'] = im
    print(len(a['images']))
    
    json.dump(a, open(fann,'w'))
