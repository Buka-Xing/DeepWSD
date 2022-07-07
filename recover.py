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
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import imageio
import DeepWSD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Parameter setting for the whole optimization.
# ref and optimized images.
ref_path   = 'images/Lena.jpg'
pred_path  = 'images/white.jpg'
# Please manually set your save path and build your folder
save_path  = './results/Optimization/White2Lena/iter%dstepsize%.6f.jpg'
final_save_path = './results/Optimization/White2Lena/final_result.jpg'

lr = 1e-2          # The step size. Small lr can better view the perceptual optimization path.
iter_num = 10000   # maximum iteration times.
decay = 1500       # decay of step size. Each time shrink 1/2. Lower bound 1e-5
output_iter = 100  # Output optimization path every setting times.

# 要使用的模型: 这里使用的是SSIM作为反向传播的模型
model = DeepWSD.DeepWSD(channels=3).to(device)

# 数据预处理, 可以定义多种变换于此
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


ref_img = Image.open(ref_path).convert("RGB")
ref = transform(ref_img).unsqueeze(0)

ref = Variable(ref.float().to(device), requires_grad=False)

Initial  = Image.open(pred_path).convert("RGB")
# pred_img = Image.open(pred_path).convert("RGB")

pred_img = Image.open(pred_path).convert("RGB")
pred = transform(pred_img).unsqueeze(0)

pred = Variable(pred.float().to(device), requires_grad=True)

model.eval()
optimizer = torch.optim.Adam([pred], lr=lr)

for i in range(iter_num + 1):
    quality_score = model(pred, ref)
    optimizer.zero_grad()
    quality_score.backward()
    optimizer.step()
    pred.data.clamp_(min=0,max=1)

    
    if i % output_iter == 0:
        pred_img = pred.squeeze().data.cpu().numpy().transpose(1, 2, 0)
        ref_img2 = ref.squeeze().data.cpu().numpy().transpose(1, 2, 0)

        fig = plt.figure(figsize=(4, 1.5), dpi=300)
        plt.subplot(131)
        plt.imshow(Initial)
        plt.title('initial', fontsize=6)
        plt.axis('off')
        plt.subplot(133)

        plt.imshow(ref_img2)
        plt.title('reference', fontsize=6)
        plt.axis('off')

        plt.subplot(132)       
        plt.imshow(np.clip(pred_img, 0, 1))

        plt.title('iter: %d \n quality_score: %.3g' % ( i, quality_score.item() ),fontsize=6)
        plt.axis('off')
        plt.savefig(save_path % (i,lr))
        plt.pause(1)
        plt.cla()
        plt.close()

    if (i+1) % decay == 0:
        lr = max(1e-5, lr * 0.5)
        optimizer = torch.optim.Adam([pred], lr=lr)

# Save the final results
imageio.imwrite(final_save_path, pred_img)

