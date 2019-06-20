
# EmptyCities for visual SLAM

[[Project]](https://bertabescos.github.io/EmptyCities/)   [[Paper]]()

Torch implementation for learning a mapping from input images that contain dynamic objects in a city environment, such as vehicles and pedestrians, to output images which are static and suitable for localization and mapping. 

<img src="imgs/CARLA.gif" width="400px" />       <img src="imgs/CITYSCAPES.gif" width="400px" />

Empty Cities: a Dynamic-Object-Invariant Space for Visual SLAM  
[Berta Bescos](https://bertabescos.github.io/), [Cesar Cadena](http://n.ethz.ch/~cesarc/), [Jose Neira](http://webdiis.unizar.es/~neira/)

## Setup

### Prerequisites
- Linux or OSX
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

### Getting Started
- Install torch and dependencies from https://github.com/torch/distro
- Install torch packages `nngraph` and `display`
```bash
luarocks install nngraph
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```
- Clone this repo:
```bash
git clone https://github.com/BertaBescos/EmptyCities_SLAM.git
cd EmptyCities_SLAM
```

### Models
Pre-trained models can de downloaded from the folder `checkpoints` in [this link](https://drive.google.com/drive/folders/1aDO7_HtVkCncGew9ZMpDJ9KCT4fYD8hm?usp=sharing). You will find a README.md file inside this folder. Place the `checkpoints` folder inside the project. 

## Test

- We encourage you to keep your data in a folder of your choice `/path/to/data/` with three subfolders `train`, `test` and `val`. The following command will run our model within all the images inside the folder `test` and keep the results in `./results/`. Images within the folder `test` should be RGB images of any size.
```bash
DATA_ROOT=/path/to/data/ name=my_name th test.lua
```
- If you prefer to feed the dynamic/static binary masks, you should concatenate it to the RGB image. We provide a python script for this on [https://github.com/bertabescos/EmptyCities](https://github.com/bertabescos/EmptyCities).
```bash
DATA_ROOT=/path/to/data/ name=my_name mask=1 th test.lua
```

## Train

- The simplest case trains only with synthetic CARLA data with G(x,m) and D(x,y,m,n). In the subfolder `/path/to/synth/data/train/` there should be the concatenated (RGB | GT | Mask) images. The utilized masks come from this simulator too, and therefore do not use the semantic segmentation model.
```bash
DATA_ROOT=/path/to/synth/data/ name=my_name th train.lua
```
- If you want to use the ORB-features-based loss you should set `lossDetector`, `lossOrientation` and `lossDescriptor` to 0 in the command line. For better adaptation to real world images it is advisable to train the model with dynamic images from a real city. These images have no groundtruth static image pair, but have groundtruth semantic segmentation. The last one is used to finetune the semantic segmentation network ERFNet for our specific goal. Real data is introduced from `epoch_synth=50` on with a probability of `pNonSynth=0.5`.
```bash
DATA_ROOT=/path/to/synth/data/ name=my_name lossDetector=1 lossOrientation=1 lossDescriptor=1 th train.lua
```
- If you want to finetune your trained model with real-world data in your training, you should set `NSYNTH_DATA_ROOT` to this dataset path.
```bash
DATA_ROOT=/path/to/synth/data/ NSYNTH_DATA_ROOT=/path/to/non/synth/data/ continue_train=1 name=my_name th train.lua
```
- (Optionally) start the display server to view results as the model trains. ( See [Display UI](#display-ui) for more details):
```bash
th -ldisplay.start 8000 0.0.0.0
```

Models are saved by default to `./checkpoints/base_512x512` (can be changed by passing `checkpoint_dir=your_dir` and `name=your_name` in options.lua).

See `options.lua` for additional training options.

## Datasets
Our synthetic dataset has been generated with [CARLA 0.8.2](https://drive.google.com/file/d/1ZtVt1AqdyGxgyTm69nzuwrOYoPUn_Dsm/view) and is available in the zipped folder `CARLA_dataset` in [this link](https://drive.google.com/drive/folders/1aDO7_HtVkCncGew9ZMpDJ9KCT4fYD8hm?usp=sharing). Information on how this dataset has been generated can be found in [here](https://github.com/bertabescos/EmptyCities).


## Citation
If you use this code for your research, please cite ours papers:

```
@article{bescos2019empty,
  title={Empty Cities: a Dynamic-Object-Invariant Space for Visual SLAM},
  author={Bescos, Berta  and Cadena, Cesar and Neira, José},
  journal={arXiv},
  year={2019}
}
```

```
@article{bescos2018empty,
  title={Empty Cities: Image Inpainting for a Dynamic-Object-Invariant Space},
  author={Bescos, Berta and Neira, José and Siegwart, Roland and Cadena, Cesar},
  journal={International Conference on Robotics and Automation (ICRA)},
  year={2018}
}
```


## Acknowledgments
Our code is heavily inspired by [pix2pix](https://github.com/phillipi/pix2pix), [DCGAN](https://github.com/soumith/dcgan.torch) and [Context-Encoder](https://github.com/pathak22/context-encoder).

# EmptyCities for visual SLAM
