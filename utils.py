# Copyright (C) <2022> Xingran Liao
# @ City University of Hong Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy of this code and
# associated documentation files (the "code"), to deal in the code without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the code,
# and to permit persons to whom the code is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the code.

# THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE Xingran Liao BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE code OR THE USE OR OTHER DEALINGS IN THE code.

#================================================
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

# Transform images to tensors. If the max size of the image is larger than 512, it will be resized to 256 due to VGG net accept 256x256
# However, for extremely large images, like max(H,W)>1000. This function will not process resize.
# Because the simple interpolation downsample will destroy image information when size is extremely large. A better downsample strategy is to use conv kernel.
def prepare_image(image, repeatNum = 1):
    H, W = image.size
    if max(H,W)>512 and max(H,W)<1000:
        image = transforms.functional.resize(image,[256,256])
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)

# Process input of VGG16 to make it close to 256
def downsample(img1, img2, maxSize = 256):
    _,channels,H,W = img1.shape
    f = int(max(1,np.round(max(H,W)/maxSize)))

    aveKernel = (torch.ones(channels,1,f,f)/f**2).to(img1.device)
    img1 = F.conv2d(img1, aveKernel, stride=f, padding = 0, groups = channels)
    img2 = F.conv2d(img2, aveKernel, stride=f, padding = 0, groups = channels)
    # For an extremely Large image, the larger window will use to increase the receptive field.
    if f >= 5:
        win = 16
    else:
        win = 4
    return img1, img2, win, f
