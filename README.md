### Setup

```
export CUDA_HOME=/usr/local/cuda-12.6/ # use yours
conda env create -f resources/environment.yml
conda activate mickey                                                                                   (base) 
```

### Download ckpts
```
wget https://storage.googleapis.com/niantic-lon-static/research/mickey/assets/mickey_weights.zip
unzip mickey_weights.zip -d weights
rm mickey_weights.zip
```

### Test
```
python demo_inference.py --im_path_ref /home/jishnu/Projects/rdd/assets/bundleSDF-rw-stanford/realworld_chair2/rgb/000000.jpg \
                         --im_path_dst /home/jishnu/Projects/rdd/assets/bundleSDF-rw-stanford/realworld_chair2/rgb/000001.jpg \
                         --intrinsics /home/jishnu/Projects/rdd/assets/bundleSDF-rw-stanford/realworld_chair2/cam_K.txt \
                         --checkpoint /home/jishnu/Projects/mickey/weights/mickey_weights/mickey.ckpt \
                         --config weights/mickey_weights/config.yaml
```