import numpy as np
import pickle
import argparse


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

    gt = pickle.load(open('labels/voc12_trainval.pkl', 'rb'))
    pls = pickle.load(open(fpl.split('.')[0] + '_t.pkl','rb'))

    c = gt.copy()
    l = len(gt)

    for i in range(l):
        bbox = np.zeros((0,4))
        label = np.zeros((0))
        for nc in range(20):  
            if not pls[i][nc].shape[0] == 0:
                pls_nc = pls[i][nc]
                bbox = np.concatenate([bbox, pls_nc[:,0:4]], 0)
                label = np.concatenate([label, [nc]*pls[i][nc].shape[0] ], 0)
        c[i]['ann']['bboxes'] = bbox.astype('float32')
        c[i]['ann']['labels'] = label.astype('int64')
        c[i]['ann']['bboxes_ignore'] = np.zeros((0,4)).astype('float32')
        c[i]['ann']['labels_ignore'] = np.zeros((0)).astype('int64')

    average_score = np.array([np.mean(np.vstack(i)[:,-1]) for i in pls])
    average_score[np.isnan(average_score)] = -1.
    num = np.sum(average_score==-1)

    score_ranking = np.argsort(-average_score)
    filters = score_ranking[0: int((score_ranking.shape[0]-num)*ratio)]

    import pickle
    c_filter = []
    for i in filters:
        if c[i]['ann']['labels'].shape[0]>0:
            c_filter.append(c[i])
    
    print(len(c_filter))
    pickle.dump(c_filter, open(fann,'wb'))
