#!/usr/bin/env bash
MVS_TRAINING="/home/khangtg/Documents/lab/mvs/dataset/mvs/dtu_dataset/train"

LOG_DIR=$1
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python train.py --logdir $LOG_DIR --dataset=dtu_yao --batch_size=1 --trainpath=$MVS_TRAINING \
                --trainlist lists/dtu/subset_train.txt --testlist lists/dtu/subset_val.txt --numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt
