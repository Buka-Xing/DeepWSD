# DeepWSD: Deep Network based Wassterstein Distance for Image Quality Assessment
----------------------------
This is the repository of paper [DeepWSD: Projecting Degradations in Perceptual Space to Wasserstein Distance in Deep Feature Space](https://arxiv.org/abs/2208.03323v1). Related Quality Assessment results and Optimization results are in `results' folder.

## Advantages of DeepWSD:
1. DeepWSD is a perceptual FR-IQA which performs excellently not only in Quality Assessment Task but also in optimization tasks.
2. DeepWSD is a complete metric which satisfy definition of metrics and may be used as a perceptual loss.
3. DeepWSD is completely free of training and can achieve nice performance on several datasets.

-----------------------------
## Updating log:
2022/10/3: Find problems on batch form DeepWSD, now substitute to original single image comparison form. Please compare images one by one.
The batch form for optimization will be released as soon as possible.

2022/8/29: Updating batch DeepWSD.

-----------------------------
## Requirements:
- imageio==2.9.0
- matplotlib==3.5.0
- numpy==1.20.1
- Pillow==9.2.0
- POT==0.8.1.0
- torch==1.8.0
- torchvision==0.9.0

------------------------------

## Useage:
1. For DeepWSD.py:
>python DeepWSD.py --ref images/Lena.jpg --dist images/white.jpg

2. For recover.py:
>python recover.py --ref_path images/Lena.jpg --pred_path images/white.jpg

3. Note 'utils.py' contains supporting function for DeepWSD, make sure they are in same folder. 

------------------------------

## Acknowledgement:
We thanks a lot for work of 'dingkeyan93' and work of DISTS. DeepWSD is mostly inspired by the insightful idea from him.  

------------------------------

## Citations:
