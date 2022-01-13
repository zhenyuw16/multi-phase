
GPUS=$1

if [ ! -d results ];then
    mkdir -p results
fi

#bash tools/dist_test.sh configs/multiphase/coco/faster_rcnn_r50_fpn_1x_coco_sup.py work_dirs/faster_rcnn_r50_fpn_1x_coco_sup/epoch_12.pth $GPUS --out results/r1.pkl 
#bash tools/dist_test.sh configs/multiphase/coco/faster_rcnn_r50_fpn_1x_coco_semi_phase1.py work_dirs/faster_rcnn_r50_fpn_1x_coco_semi_phase1/epoch_8.pth $GPUS --out results/r2.pkl
#bash tools/dist_test.sh configs/multiphase/coco/faster_rcnn_r50_fpn_1x_coco_semi_phase2.py work_dirs/faster_rcnn_r50_fpn_1x_coco_semi_phase2/epoch_7.pth $GPUS --out results/r3.pkl 

python scripts/coco/ensemble.py results/r1.pkl results/r2.pkl results/r3.pkl results/r_ensemble.pkl

python tools/analysis_tools/eval_metric.py configs/multiphase/coco/faster_rcnn_r50_fpn_1x_coco_sup.py results/r_ensemble.pkl --eval bbox

