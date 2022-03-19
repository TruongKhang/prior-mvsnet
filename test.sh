#!/usr/bin/env bash
TESTPATH="/mnt/sdb1/khang/blendedmvs"
TESTLIST="lists/blended/validation_list.txt"
CKPT_FILE=$1
python test.py --dataset=blended --batch_size=1 --testpath=$TESTPATH  --testlist=$TESTLIST --numdepth 256 --resume $CKPT_FILE --save_png --num_view 7 --max_h 576 --max_w 768 --num_stages 3 --interval_scale 1.0
