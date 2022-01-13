
GPUS=$1
RESULTNAME=$2
ANNFILE=$3

bash tools/dist_test.sh configs/multiphase/pascal_voc/faster_rcnn_r50_fpn_1x_voc07_pl.py work_dirs/faster_rcnn_r50_fpn_1x_voc07_sup/epoch_12.pth $GPUS --out $RESULTNAME --eval mAP

python scripts/pascal_voc/filter_pl.py $RESULTNAME

python scripts/pascal_voc/form_ann.py $RESULTNAME $ANNFILE 0.5
