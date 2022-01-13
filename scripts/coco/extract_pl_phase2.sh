
GPUS=$1
RESULTNAME=$2
RESULTNAME2=$3
ANNFILE=$4

bash tools/dist_test.sh configs/multiphase/coco/faster_rcnn_r50_fpn_1x_coco_pl.py  work_dirs/faster_rcnn_r50_fpn_1x_coco_semi_phase1/epoch_8.pth $GPUS --out $RESULTNAME2

python scripts/coco/pkl2json.py $RESULTNAME2

python scripts/coco/filter_pl.py $RESULTNAME2

python scripts/coco/inter.py $RESULTNAME $RESULTNAME2

python scripts/coco/form_ann.py $RESULTNAME2 $ANNFILE 1.0
