---
title: My Idea on Text Editing Methods
layout: default
parent: Text Editing 
nav_order: 1
---

다음 세가지 아이디어는 네이버 부스트캠프를 하면서 파이널 프로젝트로 후보로 두었던 **영화나 실생활에서의 이미지 내 text 탐지해서 번역하는 프로젝트** (번역한 내용을 원래의 폰트스타일에 맞게 원본 이미지에서 변형시키기)를 생각하면서 내가 정리했던 아이디어에 관한 내용이다. 

1. masking 해서 원하는 방향 text 로 주기

참고 논문) Blended Latent Diffusion 

paper: [link](https://arxiv.org/pdf/2206.02779.pdf)

code : [https://github.com/omriav/blended-latent-diffusion](https://github.com/omriav/blended-latent-diffusion)

필요한 것 : scene text editing 논문들에 쓰인 데이터셋 가지고 pretrained model 학습 

장점: 자연스러움. 다양한 스타일 custom 가능할 듯 

단점: 원본의 스타일을 많이 잃어버릴 수 있음. 영어가 아닌 다른 언어의 text를 가이드로 주는 것이 가능할지..? 이질감. 

2. masking 해서 새로운 text image를 원본 스타일로 font style transfer 해서 원본 이미지의 마스크 한 부분에 입히기. 마스킹 안한 부분과 앞의 결과물이 가이드 해서 diffusion 활용해서 채우는 것 (My Pick🌟)

필요한 것 : scene text editing 논문들에 쓰인 데이터셋 가지고 pretrained model 학습 

장점: 새로운 방법임. 배경과 text 사이 이질감이 덜해서 3번보다 자연스러울 것 같음.

단점: masking 한 부분이 통째로 있지 않아서 잘 채워질지 해봐야지 알 수 있음. 원본과 다를 수 있음. 디테일함 떨어질 수 있음. 

3. scene text removal 을 이용해서 배경 이미지를 알아낸 다음, 원본과의 차이를 이용하여 원본 text image를 알아냄. 이를 이용하여 font style transfer 하여 새로운 text image 를 배경 이미지에 입히기 

장점: 원본과 아주 유사할 수 있음. 

단점: 혹시 다른 페이퍼들이 이미 택한 방법은 아닐까 생각됨. 또한 웹툰 번역과의 차별점이 뭘까,,? 배경과 text 사이 이질감이 느껴질 수 있음. 돌아가는 느낌 기존 scene text editing 쓰는게 나음 
   
---

참고 논문 

3번 Scene Text Erasing 

- GaRnet (ECCV 2022)
    - paper: [https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760436.pdf](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760436.pdf)
    - code: [https://github.com/naver/garnet](https://github.com/naver/garnet)

2번 masking + diffusion

- RePaint ****(CVPR 2022)
    - paper : [https://arxiv.org/pdf/2201.09865.pdf](https://arxiv.org/pdf/2201.09865.pdf)
    - code : [https://github.com/andreas128/RePaint](https://github.com/andreas128/RePaint)

2,3번 Font Style Transfer 

- MXfont (ICCV 2021)
    - paper : [https://arxiv.org/pdf/2104.00887.pdf](https://arxiv.org/pdf/2104.00887.pdf)
    - code : https://github.com/clovaai/mxfont, https://github.com/clovaai/fewshot-font-generation