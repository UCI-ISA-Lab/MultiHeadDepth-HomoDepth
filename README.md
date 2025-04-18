![res](/imgs_for_repo/res.png)
# MultiHeadDepth-HomoDepth

This repo is the official implementation of [Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](https://arxiv.org/abs/2411.10013).

# Introduction
MultiHeadDepth and HomoDepth are two models proposed in ours [paper](https://arxiv.org/abs/2411.10013), targeting the well-rectified and non-rectified stereo image scenarios, respectively. Both models are designed to be lightweight and hardware-friendly, making them well-suited for AR and edge devices.

## MultiHeadDepth
**MultiHeadDepth** consists of Inverted Residual Blocks, Multi-head Cost Volumes, and Conv2DNormActivation Blocks. The _Multi-head Cost Volume_ is a newly proposed optimization based on traditional cost volumes. Instead of using cosine similarity, we adopt a combination of Layer Normalization and dot-product operations, and further introduce a multi-head mechanism. Compared to conventional cost volumes, the Multi-head Cost Volume offers improved perception capabilities and reduced inference latency.

MultiHeadDepth is suitable for well-rectified input stereo images as input, like [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and [ATD](https://www.projectaria.com/datasets/adt/) datasets.

   
<img src="/imgs_for_repo/mulh.png" height="240">

## HomoDepth
**HomoDepth** is a multi-task model for depth estimation and homography estimation, in which depth estimation depends on homography estimation. HomoDepthis is suitable for well-rectified input stereo images as input, like [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) dataset.

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

Please check Table 4. in the paper for the explanation of the notations in this table.

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

A requirement file is provided for your reference. To install the libraries, run

`$ conda create --name <env> --file requirement.txt`

## Dataset
The models are trained on [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [ATD](https://www.projectaria.com/datasets/adt/), and [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36), respectively.


