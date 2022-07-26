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
import argparse
import DeepWSD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Parameter setting for the whole optimization.
# ref and optimized images.

parser = argparse.ArgumentParser()
parser.add_argument('--ref_path',  type=str, default='images/Lena.jpg',  help='dir of the reference img')
parser.add_argument('--pred_path', type=str, default='images/white.jpg', help='dir of the initial img')

# Results Save Path, please manually set your save path and build your folder
parser.add_argument('--save_path',  type=str, default='./results/Optimization/White2Lena/iter%dstepsize%.6f.jpg',
                    help='dir of saving optimization process results ')
parser.add_argument('--final_save_path', type=str, default='./results/Optimization/White2Lena/final_result.jpg',
                    help='dir of saving final results')
# Optimization parameters
parser.add_argument('--lr',  type=float, default= 1e-2,help='The step size')
parser.add_argument('--iter_num', type=int, default=10000,help='maximum iteration times.')
parser.add_argument('--decay', type=int, default=1500,help='decay of step size. Each time shrink 1/2. Lower bound 1e-5')
parser.add_argument('--output_iter', type=int, default=100,help='Output optimization path every setting times.')
args = parser.parse_args()

model = DeepWSD.DeepWSD(channels=3).to(device)

# Data processing.
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


ref_img = Image.open(args.ref_path).convert("RGB")
ref = transform(ref_img).unsqueeze(0)

ref = Variable(ref.float().to(device), requires_grad=False)

Initial  = Image.open(args.pred_path).convert("RGB")
# pred_img = Image.open(pred_path).convert("RGB")

pred_img = Image.open(args.pred_path).convert("RGB")
pred = transform(pred_img).unsqueeze(0)

pred = Variable(pred.float().to(device), requires_grad=True)

model.eval()
optimizer = torch.optim.Adam([pred], lr=args.lr)

for i in range(args.iter_num + 1):
    quality_score = model(pred, ref)
    optimizer.zero_grad()
    quality_score.backward()
    optimizer.step()
    pred.data.clamp_(min=0,max=1)

    
    if i % args.output_iter == 0:
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
        plt.savefig(args.save_path % (i,args.lr))
        plt.pause(5)
        plt.cla()
        plt.close()

    if (i+1) % args.decay == 0:
        lr = max(1e-5, args.lr * 0.5)
        optimizer = torch.optim.Adam([pred], lr=args.lr)

# Save the final results
imageio.imwrite(args.final_save_path, pred_img)

