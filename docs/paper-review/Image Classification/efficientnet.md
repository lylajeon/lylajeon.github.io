---
title: EfficientNet Paper Review
layout: default
parent: Image Classification
grand_parent: Paper Review
nav_order: 6
---
date: 2022-10-17

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]([]())

Background : 

To get better accuracy, it was necessary to scale ConvNets' depth, width, and image resolution (which is same as image size). 



Problem we want to solve : 

Previous methods were possible to scale two or three dimensions out of depth, width, and image resolution arbitrarily, but this scaling method required tedious manual tuning and still often yielded sub-optimal accuracy and efficiency. 

We need principled method to scale up ConvNets that can achieve better accuracy and efficiency. 

![optimization](/docs/paper-review/images/efficientnet/optimization.png)

Suggesting : Compound scaling method 

It is critical to balance all dimensions of network width/depth/resolution. 

In this paper, we use compound scaling method, which uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. 



This paper empirically quantified the relationship among all three dimensions of network width, depth, and resolution.



Applied compound scaling method to MobileNets, ResNet.

Also, by using neural architecture search, this paper developed new baseline network and scaled this up to obtain a group of models called EfficientNets. 



Baseline structure 

![baseline network](/docs/paper-review/images/efficientnet/baseline_structure.png)

EfficientNets

![](./constructing_efficientnet.png)



Compound model scaling

![compound model scaling](/docs/paper-review/images/efficientnet/compound_scaling.png)



Performance

1. efficientnet 

![](/docs/paper-review/images/efficientnet/efficientnet_performance.png)

2. applying to other networks 

![](/docs/paper-review/images/efficientnet/scalingup_other.png)

FLOPS : 단위시간당 계산량 