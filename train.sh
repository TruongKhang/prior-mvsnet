python train.py --logdir checkpoints --dataset dtu_yao --batch_size 4 --trainpath /dev/dtu_dataset/train --trainlist lists/dtu/train.txt --testlist lists/dtu/val.txt --numdepth 192 --epochs 20 --ndepths 48,24,8 --depth_inter_r 4,3,1 --dlossw 1.0,1.0,1.0 --eval_freq 5
