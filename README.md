![res](/imgs_for_repo/res.png)
# MultiHeadDepth-HomoDepth

This repo is the official implementation of [Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](https://arxiv.org/abs/2411.10013)

# Introduction
MultiHeadDepth and HomoDepth are two models proposed in ours [paper](https://arxiv.org/abs/2411.10013), targeting the well-rectified and non-rectified stereo image scenarios, respectively. Both models are designed to be lightweight and hardware-friendly, making them well-suited for AR and edge devices.

## MultiHeadDepth
**MultiHeadDepth** consists of Inverted Residual Blocks, Multi-head Cost Volumes, and Conv2DNormActivation Blocks. The _Multi-head Cost Volume_ is a newly proposed optimization based on traditional cost volumes. Instead of using cosine similarity, we adopt a combination of Layer Normalization and dot-product operations, and further introduce a multi-head mechanism. Compared to conventional cost volumes, the Multi-head Cost Volume offers improved perception capabilities and reduced inference latency.


<img src="/imgs_for_repo/mulh.png" height="300">


