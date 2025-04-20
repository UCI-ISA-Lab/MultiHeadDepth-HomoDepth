import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.mobilenetv2 import InvertedResidual

import utils

class LayerNorm2D(nn.Module):
    def __init__(self, num_channels, affine=True, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            x = self.weight.view(1, self.num_channels, 1, 1) * x + self.bias.view(1, self.num_channels, 1, 1)

        return x


class MultiHeadCostVolume(nn.Module):
    def __init__(self, dim, max_disparity=10, head_size=8):
        super().__init__()
        self.norm = LayerNorm2D(dim)
        self.max_disparity = max_disparity
        self.head_size = head_size
        self.head_num = int(dim / head_size)
        self.gpconv = nn.Conv2d(max_disparity * self.head_num, max_disparity, 1,
                                groups=max_disparity)

    def forward(self, left_image, right_image):
        batch_size, _, height, width = left_image.shape
        left, right = self.norm(left_image), self.norm(right_image)
        # cost_volume = torch.zeros((batch_size, self.max_disparity * self.head_num, height, width)).to(left.device)
        left = left.view(batch_size, self.head_num, self.head_size, height, width)
        cost_list = []
        for d in range(self.max_disparity):
            right_shifted = ((torch.roll(right, shifts=d + 1, dims=3))
                             .reshape(batch_size, self.head_num, self.head_size, height, width))
            cost = torch.sum(left * right_shifted, dim=2)
            cost_list.append(cost)

        cost_volume = torch.cat(cost_list, dim=1)

        cost_volume = self.gpconv(cost_volume)
        return torch.cat((left_image, cost_volume), dim=1)


def feature_fusion(layer1, layer2):
    """
    :param layer1: the smaller layer to be upsampled double size
    :param layer2: the bigger layer
    :return: the sum up of two layers, with the size of layer2
    """
    size = (layer2.size(-2), layer2.size(-1))
    layer1 = nn.functional.interpolate(layer1, size=size,
                                       mode='bilinear', align_corners=True)
    output = layer1 + layer2

    return output

def efficient_blk(inp, oup, stride, exp_ratio=1):
    # The structure inside this block is unknown,
    # like for layer0B, how the number of channels reduce from 48 to 32 in 3 repeats
    return nn.Sequential(
        InvertedResidual(inp, inp, stride, exp_ratio),
        InvertedResidual(inp, inp, 1, exp_ratio),
        InvertedResidual(inp, oup, 1, exp_ratio))


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, freq=200):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        credit: @pierrot-lc
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels

        inv_freq = 1.0 / (freq ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, inp):
        """
        :param x: A 4d tensor of size (batch_size, ch, x, y)
        :return: Positional Encoding Matrix of size (batch_size, ch, x, y)
        """
        if len(inp.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == inp.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, org_ch, x, y = inp.shape
        pos_x = torch.arange(start=-int(x // 2), end=int(x - x // 2), step=1,
                             device=inp.device).type(self.inv_freq.type())
        pos_y = torch.arange(start=-int(y // 2), end=int(y - y // 2), step=1,
                             device=inp.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb_y = self.get_emb(sin_inp_y)
        emb = torch.zeros((org_ch, x, y), device=inp.device).type(
            inp.type()
        )

        emb[0::2, :, :] = emb_x.unsqueeze(0).repeat(y, 1, 1).permute(2, 1, 0)
        emb[1::2, :, :] = emb_y.unsqueeze(0).repeat(x, 1, 1).permute(2, 0, 1)

        self.cached_penc = emb[None, :, :, :].repeat(batch_size, 1, 1, 1)
        return self.cached_penc


class PositionalEncoding2DRec(nn.Module):
    def __init__(self, channels, freq=200):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        credit: @pierrot-lc
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels

        inv_freq = 1.0 / (freq ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -1)

    def forward(self, inp, homo):
        """
        :param tensor: A 4d tensor of size (batch_size, ch, x, y)
        :return: Positional Encoding Matrix of size (batch_size, ch, x, y)
        """
        if len(inp.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        batch_size, org_ch, x, y = inp.shape
        pos_x = torch.arange(start=-int(x // 2), end=int(x - x // 2), step=1,
                             device=inp.device).type(self.inv_freq.type())
        pos_y = torch.arange(start=-int(y // 2), end=int(y - y // 2), step=1,
                             device=inp.device).type(self.inv_freq.type())
        X, Y = torch.meshgrid(pos_x, pos_y, indexing='ij')
        points = torch.cat([X.reshape(-1), Y.reshape(-1), torch.ones_like(X.reshape(-1))]).reshape((3, -1))
        homo = homo.to(points.device)
        rotated_points = torch.matmul(homo, points)
        pos_x = rotated_points[:, 0, :]
        pos_y = rotated_points[:, 1, :]

        inv_freq = self.inv_freq.unsqueeze(0).repeat(batch_size, 1)
        sin_inp_x = pos_x.unsqueeze(2) * inv_freq.unsqueeze(1)
        sin_inp_y = pos_y.unsqueeze(2) * inv_freq.unsqueeze(1)
        l = len(self.inv_freq)
        emb_x = self.get_emb(sin_inp_x).reshape((batch_size, x, y, l * 2))
        emb_y = self.get_emb(sin_inp_y).reshape((batch_size, x, y, l * 2))
        emb = torch.zeros_like(inp)

        emb[:, 0::2, :, :] = emb_x.permute(0, 3, 1, 2)
        emb[:, 1::2, :, :] = emb_y.permute(0, 3, 1, 2)
        return emb


class MHCVP(nn.Module):
    def __init__(self, dim, max_disparity=10, head_size=8):
        super().__init__()
        self.norm = LayerNorm2D(dim)
        self.max_disparity = max_disparity
        self.head_size = head_size
        self.head_num = int(dim / head_size)
        self.gpconv = nn.Conv2d(max_disparity * self.head_num, max_disparity, 1,
                                groups=max_disparity)
        self.left_pe = PositionalEncoding2D(dim)
        self.right_pe = PositionalEncoding2DRec(dim)

    def forward(self, left_image, right_image, homo):
        batch_size, _, height, width = left_image.shape

        left, right = self.norm(left_image), self.norm(right_image)
        left = left + self.left_pe(left_image).to(left.device)
        right = right + self.right_pe(right_image, homo).to(right.device)
        left = left.reshape(batch_size, self.head_num, self.head_size, height, width)
        cost_list = []
        for d in range(self.max_disparity):
            right_shifted = (torch.roll(right, shifts=d + 1, dims=3)
                             .reshape(batch_size, self.head_num, self.head_size, height, width))
            cost = torch.sum(left * right_shifted, dim=2)
            cost_list.append(cost)
        cost_volume = torch.cat(cost_list, dim=1)

        cost_volume = self.gpconv(cost_volume)
        return torch.cat((left_image, cost_volume), dim=1)


class Stereo_MulH(nn.Module):
    def __init__(self, max_disparity=10):
        super().__init__()
        self.dmax = max_disparity
        self.layer0A = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.layer0B = efficient_blk(48 + self.dmax, 32, 1)
        self.layer0C = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))
        self.layer1A = efficient_blk(48, 40, 2)
        self.layer1B = efficient_blk(40 + self.dmax, 8, 1)
        self.layer1C = efficient_blk(8, 32, 1)
        self.layer2A = efficient_blk(40, 56, 2, 4)
        self.layer2B = efficient_blk(56 + self.dmax, 56, 1, 4)
        self.layer2C = efficient_blk(56, 8, 1, 4)
        self.layer3A = efficient_blk(56, 80, 2, 5)
        self.layer3B = efficient_blk(80 + self.dmax, 96, 1, 5)
        self.layer3C = efficient_blk(96, 56, 1, 5)
        self.layer4A = efficient_blk(80, 64, 2, 9)
        self.layer4B = efficient_blk(64 + self.dmax, 96, 1, 9)
        self.layer4C = efficient_blk(96, 96, 1, 9)
        self.layer5A = efficient_blk(64, 96, 2, 9)
        self.layer5B = efficient_blk(96 + self.dmax, 96, 1, 9)
        self.layer5C = efficient_blk(96, 96, 1, 9)

        self.layer0H = MultiHeadCostVolume(48, self.dmax)
        self.layer1H = MultiHeadCostVolume(40, self.dmax)
        self.layer2H = MultiHeadCostVolume(56, self.dmax)
        self.layer3H = MultiHeadCostVolume(80, self.dmax)
        self.layer4H = MultiHeadCostVolume(64, self.dmax)
        self.layer5H = MultiHeadCostVolume(96, self.dmax)

        self.seperated = False

    def forward(self, x):
        left, right = torch.split(x, 3, dim=1)
        stage0AL = self.layer0A(left)
        stage0AR = self.layer0A(right)
        stage0H = self.layer0H(stage0AL, stage0AR)
        stage0B = self.layer0B(stage0H)

        stage1AL = self.layer1A(stage0AL)
        stage1AR = self.layer1A(stage0AR)
        stage1H = self.layer1H(stage1AL, stage1AR)
        stage1B = self.layer1B(stage1H)

        stage2AL = self.layer2A(stage1AL)
        stage2AR = self.layer2A(stage1AR)
        stage2H = self.layer2H(stage2AL, stage2AR)
        stage2B = self.layer2B(stage2H)

        stage3AL = self.layer3A(stage2AL)
        stage3AR = self.layer3A(stage2AR)
        stage3H = self.layer3H(stage3AL, stage3AR)
        stage3B = self.layer3B(stage3H)

        stage4AL = self.layer4A(stage3AL)
        stage4AR = self.layer4A(stage3AR)
        stage4H = self.layer4H(stage4AL, stage4AR)
        stage4B = self.layer4B(stage4H)

        stage5AL = self.layer5A(stage4AL)
        stage5AR = self.layer5A(stage4AR)
        stage5H = self.layer5H(stage5AL, stage5AR)
        stage5B = self.layer5B(stage5H)

        stage5C = self.layer5C(stage5B)
        stage4C = self.layer4C(feature_fusion(stage5C, stage4B))
        stage3C = self.layer3C(feature_fusion(stage4C, stage3B))
        stage2C = self.layer2C(feature_fusion(stage3C, stage2B))
        stage1C = self.layer1C(feature_fusion(stage2C, stage1B))
        stage0C = self.layer0C(feature_fusion(stage1C, stage0B))

        return stage0C


min_m = [0.55, -0.2, -96, -0.35, 0.85, -56, 0.9]
max_m = [1.05, 0.4, -15, 0.25, 1.2, 128, 1]
class HomoDepth(nn.Module):
    def __init__(self, max_mat=max_m, min_mat=min_m, max_disparity=10, img_size=(288, 384), disp_only=False):
        super().__init__()
        width, height = img_size
        self.max_mat, self.min_mat = max_mat, min_mat
        self.dmax = max_disparity
        self.disp_only = disp_only
        self.s1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.s2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.layer0A = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.layer0B = efficient_blk(48 + self.dmax, 32, 1)
        self.layer0C = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))
        self.layer1A = efficient_blk(48, 40, 2)
        self.layer2A = efficient_blk(40, 56, 2, 4)
        self.layer3A = efficient_blk(56, 80, 2, 5)
        self.layer4A = efficient_blk(80, 64, 2, 9)
        self.layer5A = efficient_blk(64, 96, 2, 9)
        self.layer1B = efficient_blk(40 + self.dmax, 8, 1)
        self.layer1C = efficient_blk(8, 32, 1)
        self.layer2B = efficient_blk(56 + self.dmax, 56, 1, 4)
        self.layer2C = efficient_blk(56, 8, 1, 4)
        self.layer3B = efficient_blk(80 + self.dmax, 96, 1, 5)
        self.layer3C = efficient_blk(96, 56, 1, 5)
        self.layer4B = efficient_blk(64 + self.dmax, 96, 1, 9)
        self.layer4C = efficient_blk(96, 96, 1, 9)
        self.layer5B = efficient_blk(96 + self.dmax, 96, 1, 9)
        self.layer5C = efficient_blk(96, 96, 1, 9)

        self.layer0H = MHCVP(48, self.dmax)
        self.layer1H = MHCVP(40, self.dmax)
        self.layer2H = MHCVP(56, self.dmax)
        self.layer3H = MHCVP(80, self.dmax)
        self.layer4H = MHCVP(64, self.dmax)
        self.layer5H = MHCVP(96, self.dmax)

        self.bn5 = nn.BatchNorm2d(96 * 2, affine=True)
        self.layer6 = nn.Sequential(
            efficient_blk(96 * 2, 48, 1),
            nn.BatchNorm2d(48, affine=True),
            nn.ReLU6(inplace=True))
        self.layer7 = nn.Sequential(
            efficient_blk(48, 48, 1),
            nn.BatchNorm2d(48, affine=True),
            nn.ReLU6(inplace=True))
        in_dim = int(48 * height * width / 2 ** 10)
        self.fc = nn.Sequential(nn.Linear(in_dim, 1024),
                                nn.BatchNorm1d(1024, affine=True),
                                nn.ReLU6(inplace=True),
                                nn.Dropout(0.3),
                                nn.Linear(1024, 64),
                                nn.ReLU6(inplace=True),
                                nn.Linear(64, 7),
                                nn.ReLU6(inplace=True))

    def forward(self, x):
        left, right = torch.split(x, 3, dim=1)
        stage0AL = self.layer0A(left)
        stage0AR = self.layer0A(right)

        stage1AL = self.layer1A(stage0AL)
        stage1AR = self.layer1A(stage0AR)

        stage2AL = self.layer2A(stage1AL)
        stage2AR = self.layer2A(stage1AR)

        stage3AL = self.layer3A(stage2AL)
        stage3AR = self.layer3A(stage2AR)

        stage4AL = self.layer4A(stage3AL)
        stage4AR = self.layer4A(stage3AR)

        stage5AL = self.layer5A(stage4AL)
        stage5AR = self.layer5A(stage4AR)
        stage5 = torch.cat((stage5AL, stage5AR), dim=1)

        stage5 = self.bn5(stage5)
        stage6 = self.layer6(stage5)
        stage7 = self.layer7(stage6)
        homo = stage7.view(stage7.size(0), -1)
        homo_norm = self.fc(homo)

        homo_pred = self.norm2homo(homo_norm)
        homo = homo_pred
        batch_size = homo.shape[0]
        s_half = torch.tensor([[1, 1, 0.5], [1, 1, 0.5], [1, 1, 1]],
                              device=homo.device).repeat(batch_size, 1, 1)

        stage0H = self.layer0H(stage0AL, stage0AR, homo)
        stage0B = self.layer0B(stage0H)

        homo = homo * s_half
        stage1H = self.layer1H(stage1AL, stage1AR, homo)
        stage1B = self.layer1B(stage1H)

        homo = homo * s_half
        stage2H = self.layer2H(stage2AL, stage2AR, homo)
        stage2B = self.layer2B(stage2H)

        homo = homo * s_half
        stage3H = self.layer3H(stage3AL, stage3AR, homo)
        stage3B = self.layer3B(stage3H)

        homo = homo * s_half
        stage4H = self.layer4H(stage4AL, stage4AR, homo)
        stage4B = self.layer4B(stage4H)

        homo = homo * s_half
        stage5H = self.layer5H(stage5AL, stage5AR, homo)
        stage5B = self.layer5B(stage5H)

        stage5C = self.layer5C(stage5B)
        stage4C = self.layer4C(feature_fusion(stage5C, stage4B))
        stage3C = self.layer3C(feature_fusion(stage4C, stage3B))
        stage2C = self.layer2C(feature_fusion(stage3C, stage2B))
        stage1C = self.layer1C(feature_fusion(stage2C, stage1B))
        stage0C = self.layer0C(feature_fusion(stage1C, stage0B))

        if self.disp_only:
            return stage0C
        else:
            return homo_norm, homo_pred, stage0C

    def norm2homo(self, x):
        # batch_size = x.size(0)
        x = utils.norm2homo(x.detach(), self.max_mat, self.min_mat).to(x.device)
        return x
