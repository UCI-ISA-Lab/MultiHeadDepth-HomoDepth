![res](https://github.com/UCI-ISA-Lab/MultiHeadDepth-HomoDepth/blob/main/imgs_for_repo/res.pdf)
# MultiHeadDepth-HomoDepth

This repo is the official implementation of [Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](https://arxiv.org/abs/2411.10013)

# Introduction
MultiHeadDepth and HomoDepth are two models proposed in the [paper](https://arxiv.org/abs/2411.10013), targeting the well-rectified and non-rectified stereo image scenarios, respectively. Both models are designed to be lightweight and hardware-friendly, making them well-suited for AR and edge devices.

## MultiHeadDepth
MultiHeadDepth consists of Inverted Residual Blocks, Multi-head Cost Volumes, and Conv2DNormActivation Blocks. Multi-head Cost Volumes is the newly proposed optimization method based on cost volume. We replace Cosine Similarity with LayerNorm + Dot, and introduce the multi-head mechanism to the cost volume. Multi-head Cost Volume has better perception and lower latency compared with cost volume.




