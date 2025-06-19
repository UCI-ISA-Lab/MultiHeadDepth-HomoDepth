
# [CVPR 2025] MultiHeadDepth-HomoDepth

This repo is the official implementation of [Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](https://arxiv.org/abs/2411.10013), which is accepted by CVPR 2025.



# Introduction
MultiHeadDepth and HomoDepth are two models proposed in our [paper](https://arxiv.org/abs/2411.10013), targeting the well-rectified and non-rectified stereo image scenarios, respectively. Both models are designed to be lightweight and hardware-friendly, making them well-suited for AR and edge devices.

## MultiHeadDepth
**MultiHeadDepth** consists of Inverted Residual Blocks, Multi-head Cost Volumes, and Conv2DNormActivation Blocks. The _Multi-head Cost Volume_ is a newly proposed optimization based on traditional cost volumes. Instead of using cosine similarity, we adopt a combination of Layer Normalization and dot-product operations, and further introduce a multi-head mechanism. Compared to conventional cost volumes, the Multi-head Cost Volume offers improved perception capabilities and reduced inference latency.

MultiHeadDepth is suitable for well-rectified stereo images as input, like [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and [ATD](https://www.projectaria.com/datasets/adt/) datasets.

   
<img src="/imgs_for_repo/mulh.png" height="240">

## HomoDepth
**HomoDepth** is a multi-task model for depth estimation and homography estimation, in which depth estimation depends on homography estimation. HomoDepthis is suitable for non-rectified stereo images as input, like [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) dataset.

For better convergence, we made an update about the loss function of **HomoDepth**. In the paper, the loss function of homography is defined as $L_H(y,\hat{y})=|weight_w(y)-weight_w(\hat{y})|_F$. In this repo, we change it as $L_H(y,\hat{y})= |norm(y)-norm(\hat{y}) |_F$, which $norm$ is defined as `utils.homo2norm`.

![homo](/imgs_for_repo/homo_stru.png)

# Models Performances
## Main AbsRel Accuracy for MultiHeadDepth and Other SOTA Models

|                **Dataset**               | [MobileStereoNet-2D](https://arxiv.org/abs/2108.09770) |[ MobileStereoNet-3D](https://arxiv.org/abs/2108.09770) | [Dynamic-Stereo](https://dynamic-stereo.github.io/) | [Argos](https://arxiv.org/abs/2211.10551)  | MultiHeadDepth (Ours) |
|:------------------------------------:|:----------------------------------:|:----------------------------------:|:------------------------------:|:----------------------:|:---------------------------------:|
|[Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)| 0.172| 0.129     | 0.287       | 0.102      | **0.091**      |
|[ATD](https://www.projectaria.com/datasets/adt/)| 0.199   | 0.135   | 0.176   | 0.133 | **0.094**    |
|[DTU](https://roboimagedata.compute.dtu.dk/?page_id=36)| 0.147     | 0.148    | 0.339  | 0.122   | **0.101**   |

## Main Features for MultiHeadDepth and Other SOTA Models
<img src="/imgs_for_repo/scatter_plot1.png" height="420">

## HomoDepth Performance for Different Pipelines
|                                      | Argos | PreP+ Argos | PreP+ MultiheadDepth | **HomoDepth** |
|--------------------------------------|:-----:|:-----------:|:---------------------:|:-------------:|
| **AbsRel: DTU**                    | 0.122 | 0.109       | 0.099                 | **0.098**     |
| **AbsRel: Scenefolw_persp**        | 0.232 | 0.121       | 0.114                 | **0.097**     |
| **CPU Latency (ms)**               | 811.0 | 1068.3      | 884.4                 | **761.2**     |
| **GPU Latency (ms)**               | 109.0 | 312.5       | 312.6                 | **84.5**      |

Please check _Table 4._ in the paper for the explanation of the notations in this table.

## Latency on Edge Devices
<table border="1" cellspacing="0" cellpadding="5">
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="2">Orin Nano</th>
      <th colspan="2">Snapdragon</th>
    </tr>
    <tr>
      <th>CPU</th>
      <th>GPU</th>
      <th>CPU</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong><a href="https://arxiv.org/abs/2211.10551" target="_blank">Argos</a></strong></td>
      <td>8774</td>
      <td>209</td>
      <td>1424</td>
      <td>914</td>
    </tr>
    <tr>
      <td><strong>MulHeadDepth (Ours)</strong></td>
      <td><strong>6183</strong></td>
      <td>216</td>
      <td><strong>1156</strong></td>
      <td><strong>617</strong></td>
    </tr>
    <tr>
      <td><strong>HomoDepth (Ours)</strong></td>
      <td>6611</td>
      <td><strong>203</strong></td>
      <td>1512</td>
      <td>893</td>
    </tr>
  </tbody>
</table>

# Requirements
## Libraries
**Main:** python=3.10, numpy=1.26.4, pytorch=2.4.1=cuda118, scikit-image=0.24.0, imageio=2.37

A requirement file is provided for your reference. To creat the envoriment and install the libraries, run

`$ conda create --name <env> --file requirement.txt`

## Dataset
The models are trained on [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [ATD](https://www.projectaria.com/datasets/adt/), and [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36), respectively. To assist with understanding the dataset folder structure and the input format expected by the model, we provide dataloaders in `utils.py`:

Sceneflow: `SceneFlowDataset`, DTU: `DTU`, ADT: `ADT`

For ADT, we extract the frames from the ADT `.vrs` files. Please check `ADTFromOrg.extract_images` in `ADTdata.py` for more details.

# Inference
Run `infer.py` to infer a depth map with the model. Here are some examples to run it:

Test the **MultiHeadDepth** model with Sceneflow: 
```
python infer.py -l ..\data\sceneflow\flyingthings3d\frames_cleanpass\TRAIN\A\0000\left\0006.png -r ..\data\sceneflow\flyingthings3d\frames_cleanpass\TRAIN\A\0000\right\0006.png
```

Test the **HomoDepth** model with DTU, the output is saved as `.png` file with its real pixel values, and wrote to MyRes/res.png:
```
python infer.py -m HomoDepth -d DTU -f png -s MyRes/res.png -l ..\data\DTU\Rectified\scan4\rect_001_1_r5000.png -r ..\data\DTU\Rectified\scan4\rect_002_1_r5000.png 
```

# Training and Fine-tuning
We use Adam optimizer without schedulers. In the early stages of model training, the base learning rate is `1e-4`, and we select the best epoch. Afterward, we fine-tune the model with an learning rate of `4e-4` and select the optimal weights. The batch size depends on the memory of GPU. We set the batch size as 10, based on the GPU memory.

## MultiHeadDepth
We provide `train_mulh.py` for training and fine-tuning with **MultiHeadDepth**. Here are some examples to run it:

Train the model with sceneflow:
```
python train.py -p "../data/sceneflow/"
```

Train the model with DTU dataset, the number of total epochs is 1000, in which the number of validation epochs is 200. The batch size is 20, and the sample rate of the dataset is 5.
```
 python train.py -d DTU -p "../data/DTU/" -e 1000 -v 200 --batch_size 20 -sr 5
```

Based on the weights trained on ADT dataset, fine-tune the model with Middlebury dataset, set the learning rate as `1e-3`, and save the fine-tuned model weights in the folder `finetune`.
```
python train.py -d Middlebury -c .\ckpt\MulH_ADT.pt -p "../data/Middlebury/" -lr 1e-3 -s finetune 
```

## HomoDepth
We provide `train_homod.py` for training and fine-tuning with **HomoDepth**. Here are some examples to run it:

Train the HomoDepth with DTU dataset, the number of total epochs is 500, in which the number of validation epochs is 50.
```
python3 train_homod.py -p ../data/DTU/ -e 500 -v 50
```

Based on the weights trained on DTU dataset, fine-tune the model further with DTU test dataset. Set the batch size as 16, the sample rate as 5, the learning rate in training phase as `4e-4`, and the learning rate in validation phase as `1e-4`.
```
python3 train_homod.py -p ../data/DTU/ -c ./ckpt/HomoDepth_DTU.pt -sr 5 -b 16 -lr 4e-4 -lrv 1e-4
```
# Citation
```
@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Yongfan and Kwon, Hyoukjun},
    title     = {Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6252-6261}
}
```

# Contact Information
For help or issues using MultiHeadDepth and HomoDepth, please submit a GitHub issue.

For other communications related to our work, please contact Yongfan Liu (`yongfal@uci.edu`) and Hyoukjun Kwon (`hyoukjun.kwon@uci.edu`).
