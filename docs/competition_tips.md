---
title: Competition tips 
layout: default
nav_order: 4
---
date : 2022-11-16

## Using Kaggle 

1. Tidy Kaggle competitions that are similar to current task you need to solve.
2. Organize appliable ideas from many informations like Code and Discussions from corresponding competitions. 

For example, when doing object detection task for classifying recycling items, NFL Health & Safety-Helmet Assignment, VinBigData Chest X-ray Abnormalities Detection, and TensorFlow-Help Protect the Great Barrier Reef competitions can be useful. Also, writings from Code and Discussions section of the corresponding competitions which have lots of votes can be useful. 

Discussions cases

-  References 
  - Articles, Papers, ...
  - Background
- Previous competition
  - Competitions
  - Top solutions
- Best cv(cross validation) scores
  - Outlines
- Top solution
  - Closed competition
  - Idea
  - CV strategy

Code cases

- EDA
  - Sort by most votes

- Baseline
  - Sort by most votes 



Use EDA info when preprocessing

Use EDA info when applying augmentation

ex) 

When we need to preserve as many info about original image:

Flip, Rotate90, Cutout, Cutmix ... can be used. 

ColorJitter, Interpolation(Resize, Rotate30, ...), Noise, ... cannot be used. 



Applying EDA

- Pre-processing
- Augmentations
- CV Strategy
- Custom Model
- Custom Loss
- Post-processing
- ...



Other Tips 

- Most important parts 

​		CV Strategy, CV LB Correlation

- Data that can be used to training stably

  SWA(Stochastic Weight Averaging), More Ensemble, More TTA ...

- Scheduler 

  For image data, scheduler that LR decreases by every epoch. 

  At the beginning, mainly use ReduceLR or Cycle Scheduler to find appropriate epochs and learning rate.

  At last, use CosineWarmup to raise the score until the very end. 

- Cooperating method

  share mainly ideas. build baseline code for each. -> maximize the ensemble effects

- When score does not enhance

  - Go back to EDA

  - OOF analysis 

    identify if there exists common feature of the inputs that model does not predict correctly. 

    If this suceed, it can be a key to enhance the score!