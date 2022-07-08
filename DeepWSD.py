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
import os
import sys
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
import inspect
from utils import downsample
from ot.lp import wasserstein_1d

# Use L2pooling for VGG16 networks.
# Original Maxpooling will generate distortions in color channels during optimization.
class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]

        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

def ws_distance(X,Y,P=2,win=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chn_num = X.shape[1]
    X_sum = X.sum().sum()
    Y_sum = Y.sum().sum()

    X_patch   = torch.reshape(X,[win,win,chn_num,-1])
    Y_patch   = torch.reshape(Y,[win,win,chn_num,-1])
    patch_num = (X.shape[2]//win) * (X.shape[3]//win)

    X_1D = torch.reshape(X_patch,[-1,chn_num*patch_num])
    Y_1D = torch.reshape(Y_patch,[-1,chn_num*patch_num])

    X_1D_pdf = X_1D / (X_sum + 1e-6)
    Y_1D_pdf = Y_1D / (Y_sum + 1e-6)

    interval = np.arange(0, X_1D.shape[0], 1)
    all_samples = torch.from_numpy(interval).to(device).repeat([patch_num*chn_num,1]).t()

    X_pdf = X_1D * X_1D_pdf
    Y_pdf = Y_1D * Y_1D_pdf

    wsd   = wasserstein_1d(all_samples, all_samples, X_pdf, Y_pdf, P, require_sort=False)

    L2 = ((X_1D - Y_1D) ** 2).sum(dim=0)
    w  =  (1 / ( torch.sqrt(torch.exp( (- 1/(wsd+10) ))) * (wsd+10)**2))

    final = wsd + L2 * w
    # final = wsd

    return final.sum()


class DeepWSD(torch.nn.Module):

    def __init__(self, channels=3, load_weights=True):
        assert channels == 3
        super(DeepWSD, self).__init__()

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        # Rewrite the output layer of every block in the VGG network: maxpool->l2pool
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 64, 128, 256, 512, 512]

    def forward_once(self, x):
        h = x
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, as_loss=True, resize=True):
        assert x.shape == y.shape
        if resize:
            x, y, window = downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        score = 0
        layer_score=[]
        # To see score of each layer, use debugging mode of pycharm.

        for k in range(len(self.chns)):
            row_padding = round(feats0[k].size(2) / window) * window - feats0[k].size(2)
            column_padding = round(feats0[k].size(3) / window) * window - feats0[k].size(3)

            pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
            feats0_k = pad(feats0[k])
            feats1_k = pad(feats1[k])

            tmp = ws_distance(feats0_k, feats1_k, win=window)
            layer_score.append(torch.log(tmp + 1))
            score = score + tmp
        score = score / (k+1)


        if as_loss:
            return score
        else:
            # Two output strategies. Only affect PLCC.
            # For LIVE dataset the second output strategy returns higher PLCC,
            return torch.log(score + 1)**2
            # return torch.log(score + 1)

#==================================================
#  Test Part

if __name__ == '__main__':
    from PIL import Image
    import argparse
    from utils import prepare_image

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/I07.png')
    parser.add_argument('--dist', type=str, default='images/I07_09_04.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(Image.open(args.ref).convert("RGB"),resize=False).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB"),resize=False).to(device)

    model = DeepWSD().to(device)

    score = model(ref, dist, as_loss=False)
    print('score: %.4f' % score.item())
    # log    score: 5.9455
    # log**2 score: 35.3489
