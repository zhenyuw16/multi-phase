
GPUS=$1

if [ ! -d results ];then
    mkdir -p results
fi

bash tools/dist_test.sh configs/multiphase/pascal_voc/faster_rcnn_r50_fpn_1x_voc07_sup.py work_dirs/faster_rcnn_r50_fpn_1x_voc07_sup/epoch_12.pth $GPUS --out results/r1.pkl 
bash tools/dist_test.sh configs/multiphase/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_semi_phase1.py work_dirs/faster_rcnn_r50_fpn_1x_voc0712_semi_phase1/epoch_12.pth $GPUS --out results/r2.pkl
bash tools/dist_test.sh configs/multiphase/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_semi_phase2.py work_dirs/faster_rcnn_r50_fpn_1x_voc0712_semi_phase2/epoch_10.pth $GPUS --out results/r3.pkl 

python scripts/pascal_voc/ensemble.py results/r1.pkl results/r2.pkl results/r3.pkl results/r_ensemble.pkl

python tools/analysis_tools/eval_metric.py configs/multiphase/pascal_voc/faster_rcnn_r50_fpn_1x_voc07_sup.py results/r_ensemble.pkl --eval mAP

