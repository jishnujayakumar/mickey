### Setup

```shell
export CUDA_HOME=/usr/local/cuda-12.6/ # use yours
conda env create -f resources/environment.yml
conda activate mickey
```

### Download ckpts
```shell
wget https://storage.googleapis.com/niantic-lon-static/research/mickey/assets/mickey_weights.zip
unzip mickey_weights.zip -d weights
rm mickey_weights.zip
```

### Test
```shell
python demo_inference.py  --im_path_ref ./data/hrt1/realworld_chair2/rgb/000000.jpg \
                          --im_path_dst ./data/hrt1/realworld_chair2/rgb/000001.jpg \
                          --intrinsics ./data/hrt1/realworld_chair2/cam_K.mickey.txt \
                          --checkpoint ./weights/mickey_weights/mickey.ckpt \
                          --config ./weights/mickey_weights/config.yaml
```