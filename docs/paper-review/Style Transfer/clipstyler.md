---
layout: default
title : CLIPstyler 
parent: Style Transfer
grand_parent: Paper Review
nav_order : 1
---
date: 2022-10-26

[paper link](https://arxiv.org/abs/2112.00374)

<img src="https://godatadriven.com/wp-content/images/how-to-style-transfer/style-transfer-example.jpg" alt="previous" style="max-width:100%;height:auto;" />

Previously, style transfer aims to transform a content image by transferring the semantic texture of a style image. This existing neural style transfer methods require reference style images to transfer texture information of style images to content images. 

However, these methods have limitation that they require a reference style image to change the texture of the content image. 

![intro](/docs/paper-review/images/clipstyler/intro.png)

This paper proposes a new framework that enables style transfer without a reference style image, but with a text description of the desired style. 

<img src="/docs/paper-review/images/clipstyler/architecture.png" alt="arch" style="zoom:67%;" />

Content image is transformed by lightweight CNN to follow the text condition by matching the similarity betweent the CLIP model output of transferred image and the text condition. 

1. sample patches of the output image 
2. apply augmentation with different perspective views
3. obtain CLIP loss by calculating the similarity between the query text condition and the processed patches



Results

![result](/docs/paper-review/images/clipstyler/result.png)