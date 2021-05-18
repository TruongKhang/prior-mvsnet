# prior-mvsnet
### Requirements
    conda create -n venv python=3.6
    conda activate venv
    conda install pytorch==1.1.0 torchvision cudatoolkit=9.0 -c pytorch
    pip install tensorboardX
    pip install opencv-python
    pip install plyfile

Your CUDA version should be 9.0. 

    cd prior-mvsnet/MYTH
    python setup.py install

### Training
Our model is trained on DTU dataset. All training parameters are configured in file `config.json`.
To train model, first we need to prepare prior depth&confidence from `CasMVSNet`. Then, run this command to train model:

    python train.py --config config.json
    
### Testing
First, get the prior depth from [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet).
All outputs are saved in a folder `casmvs_outputs` which is placed in the test root folder.
Then, run our model to predict depth & confidence

    python test.py --dataset general_eval --batch_size 1 --testpath <your test root folder> --testlist <your dataset name> --resume <pretrained model> --outdir outputs --interval_scale 0.8 --num_view 5 --depth_scale 0.001

By default, our pretrained model is tested with number of depth planes `[80, 32, 8]` and the depth interval ratio `[4, 2, 1]`. 
You can change these parameters in file `pretrained/full/config.json`. 

Note that you need to change some parameters to fit to your dataset such as `depth_scale`, `num_view`, `max_h`, `max_w` or total of depth planes `numdepth`.
The depth scale `depth_scale` should be in milimeter (minimum depth is around 400mm).

Examples of our evaluation datasets

    # with the Family scene of Tanks&Temples dataset,
    python test.py --dataset general_eval --batch_size 1 --testpath /mnt/sdb/khang/tanksandtemples/intermediate --testlist Family --resume pretrained/full/model_best.pth --outdir outputs --interval_scale 0.8 --num_view 7 --depth_scale 0.0006 --numdepth 320 --max_h 1080 --max_w 1920
    # with DTU dataset, change #depth planes to [64, 32, 8]
    python test.py --dataset general_eval --batch_size 1 --testpath /mnt/sdb/khang/dtu_dataset/test --testlist lists/dtu/test.txt --resume pretrained/full/model_best.pth --outdir outputs --interval_scale 0.8 --num_view 7 --depth_scale 1.0 --numdepth 256 --max_h 864 --max_w 1152

