---
title: Research for the project (Image translation and style transfer)
layout: default 
parent: Scene Text Editing
nav_order: 2
---

# Scene Text Editing papers
## **SRNet**
- style retention network (SRNet)
- Editing Text in the wild. 이걸 기반해서 SwapText, STEFANN이 나옴

- Three modules
1. text conversion module : changes the text content of the source image into the target text while keeping the original text style
2. background inpainting module : erases the original text and fills the text region with appropriate texture
3. fusion module : combines the information from the two former modules, and generates the edited text images

- paper : [https://arxiv.org/pdf/1908.03047.pdf](https://arxiv.org/pdf/1908.03047.pdf)
- code : [https://github.com/endy-see/SRNet-1](https://github.com/endy-see/SRNet-1)  
    
## **STEFANN** (CVPR 2020)
- character-level text editing in image 
- the unobserved character (target) is generated from an observed character (source) being modified
- replace the source character with the generated character maintaining both geometric and visual consistency with neighboring characters

- paper : [STEFANN_CVPR_2020_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Roy_STEFANN_Scene_Text_Editor_Using_Font_Adaptive_Neural_Network_CVPR_2020_paper.pdf)
- code: [https://github.com/prasunroy/stefann](https://github.com/prasunroy/stefann)
    
## **RewriteNet** (CVPRW 2022)
- get content and style features from a text image by using scene text recognition
- rewrite new text to the original image by using the features

- paper : [https://arxiv.org/pdf/2107.11041.pdf](https://arxiv.org/pdf/2107.11041.pdf)  
- code : [https://github.com/GOGOOOMA/AIFFEL_Hackathon](https://github.com/GOGOOOMA/AIFFEL_Hackathon) (unofficial code for RewriteNet) [official github](https://github.com/clovaai/rewritenet) (currently not available)

## STRIVE (ICCV 2021)
- Scene Text Replacement in *Videos*
- the text in all frames is normalized to a frontal pose using a spatio-temporal transformer network
- the text is replaced in a single reference frame using a state-of-art still-image text replacement method
- the new text is transferred from the reference to remaining frames using a novel learned image transformation network

- paper : [G_STRIVE_Scene_Text_Replacement_in_Videos_ICCV_2021_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/G_STRIVE_Scene_Text_Replacement_in_Videos_ICCV_2021_paper.pdf)
- github : [https://github.com/striveiccv2021/STRIVE-ICCV2021](https://github.com/striveiccv2021/STRIVE-ICCV2021)

## **MOSTEL** (AAAI 2023)
- MOdifying Scene Text image at strokE Level (MOSTEL)
- generate stroke guidance maps to explicitly indicate regions to be edited
- propose a Semisupervised Hybrid Learning to train the network with both labeled synthetic images and unpaired real scene text images

- paper : [https://arxiv.org/pdf/2212.01982.pdf](https://arxiv.org/pdf/2212.01982.pdf)
- code : [https://github.com/qqqyd/MOSTEL](https://github.com/qqqyd/MOSTEL)

## SwapText (CVPR 2020)
- 하고 싶은 task 에 가장 가까운 연구임. 하지만 코드 공개가 안되어있음..
- a three-stage framework to transfer texts across scene images
- first stage: a novel text swapping network to replace text labels only in the *foreground image* 
- second stage: a background completion network to reconstruct *background images*
- third stage: the fusion network generate the *word image* by using the foreground and background images

- paper : [swaptext.pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_SwapText_Image_Based_Texts_Transfer_in_Scenes_CVPR_2020_paper.pdf)
- code : not released 


# Font style transfer (cross-language)

## FTransGAN (WACV 2021)
  - transfer font styles between different languages by observing only a few samples 
  - network into a multilevel attention form to capture both local and global features of the style images

  - paper : [Li_Few-Shot_Font_Style_Transfer_Between_Different_Languages_WACV_2021_paper.pdf](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Few-Shot_Font_Style_Transfer_Between_Different_Languages_WACV_2021_paper.pdf)
  - code : [https://github.com/ligoudaner377/font_translator_gan](https://github.com/ligoudaner377/font_translator_gan)

  
# OCR
- EasyOCR : [how to use EasyOCR](https://yunwoong.tistory.com/76), [official github](https://github.com/JaidedAI/EasyOCR), [customize EasyOCR](https://davelogs.tistory.com/94)

- Donut : [https://github.com/clovaai/donut](https://github.com/clovaai/donut)

### API
- free ocr api : [https://ocr.space/OCRAPI](https://ocr.space/OCRAPI)

# Translation
- mT5 : [https://huggingface.co/docs/transformers/model_doc/mt5](https://huggingface.co/docs/transformers/model_doc/mt5)

- mBart: [https://huggingface.co/transformers/v3.5.1/model_doc/mbart.html](https://huggingface.co/transformers/v3.5.1/model_doc/mbart.html)

- googletrans : Free and Unlimited Google translate API for Python. [Documentation](https://py-googletrans.readthedocs.io/en/latest/)

- word2word : easy-to-use word translations by Kakao Brain. [github link](https://github.com/kakaobrain/word2word)

- seq2seq : A general-purpose encoder-decoder framework for Tensorflow that can be used for Machine Translation. [github link](https://github.com/google/seq2seq)

### API

Papago : [how to use papago api](https://itadventure.tistory.com/538)

# Image Inpainting
## **RePaint** (CVPR 2022)
- A Denoising Diffusion Probabilistic Model (DDPM) based inpainting approach

- paper : [https://arxiv.org/pdf/2201.09865.pdf](https://arxiv.org/pdf/2201.09865.pdf)
- code : [https://github.com/andreas128/RePaint](https://github.com/andreas128/RePaint)

## **GARnet** (ECCV 2022)
- Scene text removal paper using Gated Attention and RoI Generation method 
- *Gated Attention* : focus on the text stroke as well as the textures and colors of the surrounding regions to remove text from the input image much more precisely
- *RoI Generation* : focus on only the region with text instead of the entire image to train the model more efficiently

- paper : [4705_ECCV_2022_paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760436.pdf)
- code : [https://github.com/naver/garnet](https://github.com/naver/garnet)
