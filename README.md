# I3D models trained on Kinetics

## Overview

Original implementation by the authors can be found in this [repository](https://github.com/deepmind/kinetics-i3d), together with details about the pre-processing techniques.


## Running the code

### Setup
clone this repository using

$ git clone https://github.com/andrewatef/kinetics-i3d

## Create list files
We use list files in ```data/ucf101/``` subdir to make the code find RGB images and flow data saved on disk. You have to adapt the list files to make sure the list files contain the right path to your data. Specifically, for RGB data, you have to update ```data/ucf101/rgb.txt```. Each line in in this file should be in the format:

```dir_name_of_imgs_of_a_video /path/to/img_dir num_imgs label```
For example, if your RGB data of UCF101 is saved in '/data/user/ucf101/rgb', and there are 13320 subdirs in this folder, each subdir contains images from a video. If in subdir v_BalanceBeam_g14_c02, there are 96 images, and the ground truth of this video is 4, then the line for this subdir is:

```v_BalanceBeam_g14_c02 /data/user/ucf101/rgb/v_BalanceBeam_g14_c02 96 4```
Similarly, update data/ucf101/flow.txt for flow data. Note: we use one file to include x and y part of flow data, so we use {：s} in each line to placehold x or y in the data path. For example, if your flow data are placed like this:
```
|---tvl1_flow
|   |---x
|   |--- y
```
then you can write each line in flow.txt like this:

``` v_Archery_g01_c06 /data4/zhouhao/dataset/ucf101/tvl1_flow/{:s}/v_Archery_g01_c06 107 2 ```
i.e, use {:s} replace x or y in path. If you are confused, please refer our code to see data loading details.

## Train on UCF101 on RGB data and flow data
### Finetune on split1 of RGB data of UCF101
```
CUDA_VISIBLE_DEVICES=0 python finetune.py ucf101 rgb 1
```
### Finetune on split2 of flow data of UCF101
```
CUDA_VISIBLE_DEVICES=0 python finetune.py ucf101 flow 2
```
We share our trained models on UCF101(RGB & FLOW) in GoogleDrive and BaiduDisk (password:ddar). You can download these models and put them in model folder of this repo. In this way you can skip the train commands above and directly run test in the next step.

# Requirements

```
pip install -r requirements.txt
```
Sonnet](https://github.com/deepmind/sonnet).


### Sample code

Run the example code using

`$ python evaluate_sample.py`

With default flags, this builds the I3D two-stream model, loads pre-trained I3D
checkpoints into the TensorFlow session, and then passes an example video
through the model. The example video has been preprocessed, with RGB
NumPy arrays provided (see more details below).

The script outputs the top 5 Kinetics classes predicted by the model 
with their probability. Using the default flags, the output should 
resemble the following up to differences in numerical precision:

```
Norm of logits: 76.697174

Top 5 classes and probabilities
playing cricket        100.00%
shooting goal (soccer) 0.00%
hurling (sport)        0.00%
catching or throwing softball 0.00%
catching or throwing baseball 0.00%
```

### Running the test

The test file can be run using

`$ python i3d_test.py`

This checks that the model can be built correctly and produces correct shapes.

## Further details

### Provided checkpoints

The default model has been pre-trained on ImageNet and then Kinetics; other
flags allow for loading a model pre-trained only on Kinetics and for selecting
only the RGB or Flow stream. The script `multi_evaluate.sh` shows how to run all
these combinations, generating the sample output in the `out/` directory.

The directory `data/checkpoints` contains the four checkpoints that were
trained. The ones just trained on Kinetics are initialized using the default
Sonnet / TensorFlow initializers, while the ones pre-trained on ImageNet are
initialized by bootstrapping the filters from a 2D Inception-v1 model into 3D,
as described in the paper.

The models are trained using the training split of Kinetics. On the Kinetics
test set, we obtain the following top-1 / top-5 accuracy:

Model          | ImageNet + Kinetics | Kinetics
-------------- | :-----------------: | -----------
RGB-I3D        | 71.1 / 89.3         | 68.4 / 88.0
Flow-I3D       | 63.4 / 84.9         | 61.5 / 83.4
Two-Stream I3D | 74.2 / 91.3         | 71.6 / 90.0

# Sample data and preprocessing

## preprocessing

The preprocessing file can be run using

`$ python preprocessing.py --video_path "Path_to the Video"  `

Ex: 

 `$ python preprocessing.py --video_path "VID_NaderSquat.mp4" `


This will convert the video to a numpy array in the data/ directory 
with the name "VID_NaderSquat_rgb.npy"

## Data

Under data/ there are 3 input videos.
- cricket.avi
- VID_NaderRun.mp4
- VID_MultiAction.mp4

And also cricket_rgb.npy

## Test with your own video

`$ python evaluate_sample.py --video_path "Path_to the Video" `

## Our testing results on Multia-ction videos collected from different data-sets

[Output_videos](https://github.com/andrewatef/kinetics-i3d/tree/master/output_videos)

### Model accuracy on multi-action videos

![Model accuracy](https://github.com/andrewatef/kinetics-i3d/blob/master/updated_Graph.jpeg)

 
