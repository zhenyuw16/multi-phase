# Data-Uncertainty Guided Multi-Phase Learning for Semi-supervised Object Detection

This is the mmdetection implementation of our CVPR2021 paper:

>Zhenyu Wang, Yali Li, Ye Guo, Lu Fang, Shengjin Wang. Data-Uncertainty Guided Multi-Phase Learning for Semi-supervised Object Detection. [ArXiv](https://arxiv.org/abs/2103.16368).

# Installation

This code is based on mmdetection v2.18.
Please install the code according to the [mmdetection step](https://github.com/open-mmlab/mmdetection/blob/v2.18.0/docs/get_started.md) first.
Run:
```bash
pip install ensemble_boxes
```
to prepare for ensembling the results.

### data preparation

```bash
multiphase
├──data
|  ├──VOCdevkit
|  |  ├──VOC2007
|  |  ├──VOC2012
|  ├──coco
|  |  ├──annotations
|  |  |  ├──instances_train2014.json
|  |  |  ├──instances_valminusminival2014.json
|  |  |  ├──instances_minival2014.json
|  |  ├──images
|  |  |  ├──train2014
|  |  |  ├──val2014
```

# Running scripts

## pascal voc

Run:
```bash
python tools/dataset_converters/pascal_voc.py data/VOCdevkit -o labels
```
to prepare the dataset.
Then, to train the supervised model, run (the default gpu number for VOC is 4):
```bash
bash tools/dist_train.sh configs/multiphase/pascal_voc/faster_rcnn_r50_fpn_1x_voc07_sup.py 4
```
With the supervised model, generating pseudo labels for the first phase:
```bash
bash scripts/pascal_voc/extract_pl_phase1.sh 4 labels/rvoc.pkl labels/voc12_trainval_pl_phase1.pkl 
```
Then, perform semi-supervised learning for the first phase:
```bash
bash tools/dist_train.sh configs/multiphase/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_semi_phase1.py 4
```
Generating pseudo labels for the second phase:
```bash
bash scripts/pascal_voc/extract_pl_phase2.sh 4 labels/rvoc.pkl labels/rvoc2.pkl labels/voc12_trainval_pl_phase2.pkl
```
Semi-supervised learning for the second phase:
```bash
bash tools/dist_train.sh configs/multiphase/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_semi_phase2.py 4
```
Finally, model ensemble for the detection results:
```bash
bash scripts/pascal_voc/ensemble_test.sh 4
```

## coco
For the COCO dataset, the basic pipieline is the same, the default gpu number is 8:
```bash
bash tools/dist_train.sh configs/multiphase/coco/faster_rcnn_r50_fpn_1x_coco_sup.py 8
bash scripts/coco/extract_pl_phase1.sh 8 labels/rvcoco.pkl labels/coco115k_trainval_pl_phase1.json 
bash tools/dist_train.sh configs/multiphase/coco/faster_rcnn_r50_fpn_1x_coco_semi_phase1.py 8
bash scripts/coco/extract_pl_phase2.sh 8 labels/rvcoco.pkl labels/rvcoco2.pkl labels/coco115k_trainval_pl_phase2.json
bash tools/dist_train.sh configs/multiphase/coco/faster_rcnn_r50_fpn_1x_coco_semi_phase2.py 8
bash scripts/coco/ensemble_test.sh 8
'''

### Citation

If you find this repo useful for your research, please consider citing the paper as follows:

```
### Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```
@inproceedings{wang2021data,
  title={Data-uncertainty guided multi-phase learning for semi-supervised object detection},
  author={Wang, Zhenyu and Li, Yali and Guo, Ye and Fang, Lu and Wang, Shengjin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
Contact us for any questions.
