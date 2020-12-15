#!/usr/bin/env bash
TESTPATH="/home/khangtg/Documents/lab/mvs/dataset/mvs/reconstructability_dataset/simulation"
TESTLIST="lists/simulation/test.txt"
CKPT_FILE=$1
python test.py --dataset=general_eval --batch_size=1 --testpath=$TESTPATH  --testlist=$TESTLIST --loadckpt $CKPT_FILE ${@:2}
