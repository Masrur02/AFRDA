# AFRDA: Attentive Feature Refinement for Domain Adaptive Semantic Segmentation
<div style="display:flex; justify-content:center; align-items:center; width:100%; margin:0 auto;">
  <a href="https://ieeexplore.ieee.org/document/11115028"><strong>Paper (IEEE)</strong></a>

<div style="display:flex; justify-content:center; align-items:center; width:100%; margin:50 auto;">
  <a href="https://ieeexplore.ieee.org/document/11115028"><strong>Paper</strong></a>
  <span style="width:20px;"></span>
  <a href="https://arxiv.org/abs/2507.17957"><strong>arXiv</strong></a>
</div>

## Environment Setup

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/mic-seg
source ~/venv/mic-seg/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Please, download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `AFRDA/`.

## Dataset Setup

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia (Optional):** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.


**MESH**: You can collect your own forest environment dataset and put them to `data/MESH`. 

The final folder structure should look like this:
```bash 
DEDA
├── ...
├── data
│   
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│  
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── rugd
│   │   ├── images
│   │   ├── labels
│   ├── MESH
│   │   ├── images
│   │   ├── labels
│   │ 
├── 
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```
# Training
A training job for gta2cs can be launched using:
```bash
python run_experiments.py --config configs/afr/gtaHR2csHR_afr_hrda.py
```
A training job for syn2cs can be launched using:
```bash
python run_experiments.py --config configs/afr/synHR2csHR_hrda.py
```
and a training job for rugd2mesh can be launched using:
```bash
python run_experiments.py --config configs/afr/rugd2mesh_hrda.py
```
The logs and checkpoints are stored in 
```bash 
work_dirs/
```
## Evaluation

A trained model can be evaluated using:

```shell
sh test.sh work_dirs/run_name/
```

The predictions are saved for inspection to
`work_dirs/run_name/preds`
and the mIoU of the model is printed to the console.

When training a model on Synthia→Cityscapes, please note that the
evaluation script calculates the mIoU for all 19 Cityscapes classes. However,
Synthia contains only labels for 16 of these classes. Therefore, it is a common
practice in UDA to report the mIoU for Synthia→Cityscapes only on these 16
classes. As the Iou for the 3 missing classes is 0, you can do the conversion
`mIoU16 = mIoU19 * 19 / 16`.

## Checkpoints

Below, we provide checkpoints of AFRDA for the different benchmarks.

* [AFRDA for GTA→Cityscapes](https://drive.google.com/file/d/14tnErcJmUShyQtzmTBLZ9BG_cRvGM4vV/view?usp=sharing)
* [AFRDA for Synthia→Cityscapes](https://drive.google.com/file/d/103tKPC72d_FNjyJcAromFBCGpoCfhxZM/view?usp=sharing)
* [AFRDA for RUGD→MESH](https://drive.google.com/file/d/1Iwm1Whm5hnxhAzv7mfvDRnHN5o58Ubnn/view?usp=sharing)

The checkpoints come with the training logs. Please note that: The logs provide the mIoU for 19 classes. For Synthia→Cityscapes, it is necessary to convert the mIoU to the 16 valid classes. Please, read the section above for converting the mIoU.

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for AFRDA are:

* [configs/afr/gtaHR2csHR_afr_hrda.py](configs/afr/gtaHR2csHR_afr_hrda.py):
  Annotated config file for AFR on GTA→Cityscapes.
* [configs/afr/rugd2meshHR_afr_hrda.py](configs/afr/rugdHR2meshHR_afr_hrda.py):
  Annotated config file for AFR on RUHD→MESH.
* [experiments.py](experiments.py):
  Definition of the experiment configurations in the paper.
* [mmseg/models/decode_heads/hrda_head.py](mmseg/models/decode_models/hrda_head.py):
  Implementation of the hrda head with integrated AFR module.
* [mmseg/models/uda/dacs.py](mmseg/models/uda/dacs.py):
  Implementation of the DAFormer/HRDA self-training.
* [tools/inf_ros.py](tools/inf_ros.py):
  Inference code for implementation in ROS.
* [in_ros.sh](in_ros.sh):
  bash file for running the inference code with ros.
## Deployment
For navigating we integrate AFRDA with [POVNav](https://github.com/Dpushp/POVNav). And then we deploy it on a Clearpath Husky Robot. We assume [ros-noetic]([https://github.com/lhoyer/MIC](https://wiki.ros.org/noetic/Installation/Ubuntu)), the anaconda environment mentioned in the Environment Setup, Husky Robot's sensor and base workspace is already on your Robot's onboard computer.

### Navigation Instruction (In order)
1. Open a terminal
```shell
roscore
```
2. In another terminal: run the RGB and the localization launch file
 ```shell 
cd husky_n_sensors
source devel/setup.bash
roslaunch realsense2_camera rs_d400_and_t265.launch

```
3. In another terminal run the AFRDA (Download the checkpoint for the forest environment and put it in the work_dirs/local-basic folder)
```shell 
cd AFRDA
Conda activate AFRDA
sh in_ros.sh AFRDA_mesh

```
4. In another terminal use the following command to connect the joystick
```shell 
sudo ds4drv

```
5. In another terminal run the husky base launch file:
```shell 
cd husky_ws
source devel/setup.bash
roslaunch husky_base base.launch
```
6. Now download the PovNav and in another terminal
```shell 
cd povnav_ws
catkin_make
source devel/setup.bash
roslaunch pov_nav sim_pov_nav.launch
```
7. Open rviz, visualize the necessary topics, and from 2D Nav goal option give goal to the planner
8. Now use a python code to publish the velocities generated from the PovNav to Husky



## Acknowledgements

AFRDA is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [HRDA](https://github.com/lhoyer/HRDA)
* [MIC](https://github.com/lhoyer/MIC)
* [POVNav](https://github.com/Dpushp/POVNav)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
