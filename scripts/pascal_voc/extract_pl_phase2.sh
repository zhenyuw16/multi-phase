
GPUS=$1
RESULTNAME=$2
RESULTNAME2=$3
ANNFILE=$4

bash tools/dist_test.sh configs/multiphase/pascal_voc/faster_rcnn_r50_fpn_1x_voc07_pl.py work_dirs/faster_rcnn_r50_fpn_1x_voc0712_semi_phase1/epoch_12.pth $GPUS --out $RESULTNAME2 --eval mAP

python scripts/pascal_voc/filter_pl.py $RESULTNAME2

python scripts/pascal_voc/inter.py $RESULTNAME $RESULTNAME2

python scripts/pascal_voc/form_ann.py $RESULTNAME2 $ANNFILE 1.0
