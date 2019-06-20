
# EmptyCities for visual SLAM

[[Project]](https://bertabescos.github.io/EmptyCities/)   [[Paper]]()

Torch implementation for learning a mapping from input images that contain dynamic objects in a city environment, such as vehicles and pedestrians, to output images which are static and suitable for localization and mapping. 

<img src="imgs/CARLA.gif" width="400px" /> <img src="imgs/CITYSCAPES.gif" width="400px" />

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
git clone git@github.com:BertaBescos/EmptyCities_SLAM.git
cd EmptyCities_SLAM
```

### Models
Pre-trained models can de downloaded from the folder `checkpoints` in [this link](https://drive.google.com/file/d/1Va03vxrtI2W671ERxjPPows1uzRKGNKF/view?usp=sharing). Place the `checkpoints` folder inside the project.

## Test

- We encourage you to keep your data in a folder of your choice `/path/to/data/` with three subfolders `train`, `test` and `val`. The following command will run our model within all the images inside the folder `test` and keep the results in `./results/mGAN`. Images within the folder `test` should be RGB images of any size.
```bash
DATA_ROOT=/path/to/data/ th test.lua
```
For example:
```bash
DATA_ROOT=/imgs/ th test.lua
```
- If you prefer to feed the dynamic/static binary masks, you should concatenate it to the RGB image. We provide a python script for this.
```bash
DATA_ROOT=/path/to/data/ mask=1 th test.lua
```
- Finally, if the groundtruth images are available you should concatenate them too (RGB | GT | Mask).
```bash
DATA_ROOT=/path/to/data/ mask=1 target=1 th test.lua
```
The test results will be saved to an html file here: `./results/mGAN/latest_net_G_val/index.html`.


## Train

- The simplest case trains only with synthetic CARLA data. In the subfolder `/path/to/synth/data/train/` there should be the concatenated (RGB | GT | Mask) images. The utilized masks come from this simulator too, and therefore do not use the semantic segmentation model.
```bash
DATA_ROOT=/path/to/synth/data/ th train.lua
```
- For better adaptation to real world images it is advisable to train the model with dynamic images from a real city. These images have no groundtruth static image pair, but have groundtruth semantic segmentation. The last one is used to finetune the semantic segmentation network ERFNet for our specific goal. Real data is introduced from `epoch_synth=50` on with a probability of `pNonSynth=0.5`.
```bash
DATA_ROOT=/path/to/synth/data/ NSYTNH_DATA_ROOT=/path/to/real/data/ epoch_synth=50 pNonSynth=0.5 th train.lua
```
- (CPU only) The same training command without using a GPU or CUDNN. Setting the environment variables `gpu=0 cudnn=0` forces CPU only
```bash
DATA_ROOT=/path/to/synth/data/ gpu=0 cudnn=0 th train.lua
```
- (Optionally) start the display server to view results as the model trains. ( See [Display UI](#display-ui) for more details):
```bash
th -ldisplay.start 8000 0.0.0.0
```

Models are saved by default to `./checkpoints/mGAN` (can be changed by passing `checkpoint_dir=your_dir` and `name=your_name` in train.lua).

See `opt` in train.lua for additional training options.

## Datasets
Our synthetic dataset has been generated with [CARLA 0.8.2](https://drive.google.com/file/d/1ZtVt1AqdyGxgyTm69nzuwrOYoPUn_Dsm/view). Within our folder `/scripts/CARLA` we provide some python and bash scripts to generate the paired images. The files `/scripts/CARLA/client_example_read.py` and `/scripts/CARLA/client_example_write.py` shoule be run instead of the `/PythonClient/client_example.py` provided in CARLA_0.8.2. Images with different weather conditions should be generated. 
- The following bash scripts store the images with dynamic objects in `path/to/dataset/`, as well as the control inputs of the driving car and the trajectory that has been followed in `Control.txt` and `Trajectory.txt` respectively. CARLA provides two different towns setups: Town01 has been used for generating the training and validations sets, and Town02 for the testing set. 
```bash
bash scripts/CARLA/CreateDynamicDatasetTown01.sh path/to/my/folder/
```
```bash
bash scripts/CARLA/CreateDynamicDatasetTown02.sh path/to/my/folder/
```
- These scripts read the previous stored `Control.txt` files and try to replicate the same trajectories in the same scenarios with no dynamic objects. The followed trajectory and the one in `Trajectory.txt` are compared to check that the vehicle position is kept the same.
```bash
bash scripts/CARLA/CreateStaticDatasetTown01.sh path/to/my/folder/
```
```bash
bash scripts/CARLA/CreateStaticDatasetTown02.sh path/to/my/folder/
```
- Once all these images are generated, the dynamic images should be stored together in a folder with the subfolders `/train/`, `/test/` and `/val/`. The same for the static images and for the dynamic/static binary masks. We provide the following bash script:
```bash
bash scripts/CARLA/setup.sh path/to/my/folder/ path/to/output/
```
For better adaptation to real world images, we have used the [Cityscapes dataset](https://www.cityscapes-dataset.com/).

## Setup Training/Validation/Test data
### Generating Pairs
- We provide a python script to generate the CARLA training, validation and test data in the needed format. The following script concatenates the images {A,B,C} where A is the image with dynamic objects, B is the groundtruth static image, and C is the dynamic/static binary mask. 
```bash
python scripts/setup/combineCARLA.py --fold_A /path/to/output/A/ --fold_B /path/to/output/B/ --fold_C /path/to/output/C/ --fold_ABC /path/to/output/ABC/
```
- Also, to format the Cityscapes images we provide the following python script. You should run it for the `/val` folder too.
```bash
bash scripts/setup/combineCITYSCAPES.py --fold_A /path/to/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train --fold_B /path/to/CITYSCAPES/gtFine_trainvaltest/gtFine/train --fold_AB /path/to/output/train
```
**Further notes**: We provide a small dataset within `/datasets` as an example. For a good performance the training dataset should consist of many more images.

## Display UI
Optionally, for displaying images during training and test, use the [display package](https://github.com/szym/display).

- Install it with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Then start the server with: `th -ldisplay.start`
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)

By default, the server listens on localhost. Pass `0.0.0.0` to allow external connections on any interface:
```bash
th -ldisplay.start 8000 0.0.0.0
```
Then open `http://(hostname):(port)/` in your browser to load the remote desktop.

L1 error is plotted to the display by default. Set the environment variable `display_plot` to a comma-seperated list of values `errL1`, `errG` and `errD` to visualize the L1, generator, and descriminator error respectively. For example, to plot only the generator and descriminator errors to the display instead of the default L1 error, set `display_plot="errG,errD"`.

## Citation
If you use this code for your research, please cite our paper Empty Cities: Image Inpainting for a Dynamic Objects Invariant Space</a>:

```
@article{bescos2018empty,
  title={Empty Cities: Image Inpainting for a Dynamic-Object-Invariant Space},
  author={Bescos, Berta and Neira, José and Siegwart, Roland and Cadena, Cesar},
  journal={arXiv},
  year={2018}
}
```

## Acknowledgments
Our code is heavily inspired by [pix2pix](https://github.com/phillipi/pix2pix), [DCGAN](https://github.com/soumith/dcgan.torch) and [Context-Encoder](https://github.com/pathak22/context-encoder).

# EmptyCities
