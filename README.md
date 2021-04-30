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

### How to use 
First, get the prior depth from [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet).
All outputs are saved in a folder `casmvs_outputs` which is placed in the test root folder.
Then, run our model to predict depth & confidence

    python test.py --dataset general_eval --batch_size 1 --testpath <your test root folder> --testlist <your dataset name> --resume <pretrained model> --outdir outputs --interval_scale 0.8 --num_view 5 --depth_scale 0.001

Note that you need to change some parameters to fit to your dataset such as `depth_scale`, `num_view`, `max_h`, `max_w`... 
The depth scale should be in milimeter (minimum depth is around 400mm).
