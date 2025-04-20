import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import imageio.v2 as iio
from skimage.transform import resize
import re
from collections import OrderedDict


def readPFM(file):
    """
    :param file:
    :return: Tuple(ndarray(m, n)), float32
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def img2tensor(path, size=(288, 384)):
    """
    read image from drive and load as tensor
    :param path:
    :param size: tuple, len = 2
    :return:
    """
    image = iio.imread(path)
    image = resize(image, size, anti_aliasing=False, order=0)
    if len(image.shape) == 3:
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).float()
    elif len(image.shape) == 2:
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0).repeat((3, 1, 1)).float()


def gard_map(x):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(x.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(x.device)
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    grad_sq = torch.clamp(grad_x ** 2 + grad_y ** 2, min=1e-4)
    # Limit the min value to prevent grad_sq generate nan
    grad = torch.sqrt(grad_sq)
    return grad


def remove_module_prefix(in_dict):
    """
    nn.DataParallel will warp the key value of the
    state_dict with "module" prefix
    This function remove it to load it into the model
    """
    old_state_dict = in_dict['model_state_dict']
    new_state_dict = OrderedDict()
    flag = False
    for k, v in old_state_dict.items():
        if k.startswith('module.'):
            flag = True
            new_state_dict[k.replace('module.', '')] = v
    if flag:
        in_dict['model_state_dict'] = new_state_dict
    return in_dict


class SmoothLoss(nn.Module):
    def __init__(self, lambd, level=5):
        super().__init__()
        self.lambd = lambd
        self.l = level

    def forward(self, pred, gt):
        loss = F.smooth_l1_loss(pred, gt)
        sub_pred, sub_gt = pred, gt
        for i in range(self.l):
            loss += self.lambd * F.smooth_l1_loss(gard_map(sub_pred), gard_map(sub_gt))
            sub_pred = F.interpolate(sub_pred, scale_factor=0.5, mode='bilinear')
            sub_gt = F.interpolate(sub_gt, scale_factor=0.5, mode='bilinear')

        return loss


def abs_rel_error(pred_depth, gt_depth):
    mask = torch.logical_and(gt_depth > 0,
                             torch.logical_not(torch.isinf(gt_depth)) & torch.logical_not(torch.isnan(gt_depth)))
    abs_rel = torch.abs(pred_depth - gt_depth) / gt_depth
    abs_rel = abs_rel[mask]
    return abs_rel.mean()


def D1_metric(pred_depth, gt_depth, threshold=0.05):
    mask = torch.logical_and(gt_depth > 0,
                             torch.logical_not(torch.isinf(gt_depth)) & torch.logical_not(torch.isnan(gt_depth)))
    pred_depth, gt_depth = pred_depth[mask], gt_depth[mask]
    E = torch.abs(pred_depth - gt_depth)
    err_mask = (E > 0) & (E / gt_depth.abs() > threshold)
    return torch.mean(err_mask.float())


def RMSE(pred_depth, gt_depth):
    mask = torch.logical_and(gt_depth > 0,
                             torch.logical_not(torch.isinf(gt_depth)) & torch.logical_not(torch.isnan(gt_depth)))
    mse = F.mse_loss(pred_depth[mask], gt_depth[mask])
    return torch.sqrt(mse)


class WMSELoss(torch.nn.Module):
    def __init__(self, wight, device):
        super().__init__()
        self.weight = torch.tensor([[wight, wight, 1], [wight, wight, 1], [1, 1, 1]],
                                   dtype=torch.float32, requires_grad=False,
                                   device=device)

    def forward(self, inp, target):
        inp = inp * self.weight
        target = target * self.weight
        loss = torch.mean((inp - target) ** 2)
        return loss


def homo2norm(x, max_mat, min_mat):
    batch_size = x.size(0)
    homo = torch.zeros((batch_size, 7))
    res = torch.zeros((batch_size, 7))

    tmp = x[:, :2, :]
    homo[:, :6] = tmp.reshape((-1, 6))
    homo[:, 6] = x[:, 2, 2]

    for i in range(7):
        res[:, i] = (homo[:, i] - min_mat[i]) / (max_mat[i] - min_mat[i]) * 6

    return res


def norm2homo(x, max_mat, min_mat):
    batch_size = x.size(0)
    homo = torch.zeros((batch_size, 7))
    res = torch.zeros((batch_size, 3, 3))

    for i in range(7):
        homo[:, i] = x[:, i] / 6 * (max_mat[i] - min_mat[i]) + min_mat[i]
    tmp = homo[:, :6]
    res[:, :2, :] = tmp.reshape(batch_size, 2, 3)
    res[:, 2, 2] = homo[:, 6]

    return res


class Middlebury(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list, self.disp_list = self.get_file_list()

    def get_file_list(self):
        image_list, disp_list = [], []
        image_dir = os.path.join(self.root_dir, 'MiddEval3-data-Q/MiddEval3/trainingQ')
        disp_dir = os.path.join(self.root_dir, 'MiddEval3-GT0-Q/MiddEval3/trainingQ')
        for file in os.listdir(image_dir):
            image_list.append(os.path.join(image_dir, file))
            disp_list.append(os.path.join(disp_dir, file, 'disp0GT.pfm'))
        return image_list, disp_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        disparity_path = self.disp_list[idx]

        left_dir = os.path.join(image_path, 'im0.png')
        left_image = img2tensor(left_dir)
        right_dir = os.path.join(image_path, 'im1.png')
        right_image = img2tensor(right_dir)
        image = torch.cat((left_image, right_image), dim=0)
        disparity = readPFM(disparity_path)
        disparity = resize(disparity, (288, 384), anti_aliasing=False, order=0)
        disparity = torch.tensor(disparity, dtype=torch.float32).unsqueeze(0).float()
        mask = torch.logical_or(disparity < 0,
                                torch.isinf(disparity) | torch.isnan(disparity))
        disparity[mask] = 0

        return image, disparity


class SceneFlowDataset(Dataset):
    def __init__(self, root_dir, train=True, stereo=True):
        self.root_dir = root_dir
        self.classes = os.listdir(self.root_dir)
        self.train = train
        self.stereo = stereo
        self.image_list = self.get_file_list(0.7)

    def get_file_list(self, sp):
        image_list = []
        for class_name in self.classes:
            class_list = []
            left_dir = os.path.join(self.root_dir, class_name, 'frames_cleanpass')
            for dirpath, _, filenames in os.walk(left_dir):
                if 'left' in dirpath:
                    for filename in filenames:
                        if filename.endswith('.png'):
                            image_path = os.path.join(dirpath, filename)
                            rel_path = os.path.relpath(image_path, left_dir)
                            image_name = os.path.basename(image_path)
                            sub_path = rel_path[:-9]
                            disparity_path = os.path.join(self.root_dir, class_name, 'disparity', sub_path,
                                                          os.path.basename(image_path)[:-4] + '.pfm')
                            class_list.append((image_path, disparity_path))

            split_index = int(len(class_list) * sp)
            if self.train:
                image_list += class_list[:split_index]
            else:
                image_list += class_list[split_index:]

        print(f"{len(image_list)} data collected")
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path, disparity_path = self.image_list[idx]

        left_image = img2tensor(image_path)
        disparity = readPFM(disparity_path)
        disparity = resize(disparity, (288, 384), anti_aliasing=False, order=0)
        disparity = torch.tensor(disparity, dtype=torch.float32).unsqueeze(0).float()

        if self.stereo:
            right_dir = os.path.join(os.path.dirname(image_path)[:-5], 'right')
            right_image_path = os.path.join(right_dir, os.path.basename(image_path))
            right_image = img2tensor(right_image_path)
            left_image = torch.cat((left_image, right_image), dim=0)

        return left_image, disparity

    def disparity2depth(self, disparity):
        return 1050 / disparity


class DTU(Dataset):
    def __init__(self, root_dir, train='train', subset=False, output_homo=True):
        self.root_dir = root_dir
        self.classes = os.listdir(self.root_dir)
        self.output_homo = output_homo
        # The set selections are the same as MVSNet
        if subset:
            training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44]

            validation_set = [3, 5, 17, 21, 28]
            evaluation_set = [1, 4, 9, 10, 11]
        else:
            training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                            45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                            74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                            101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                            121, 122, 123, 124, 125, 126, 127, 128]
            validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
            evaluation_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77,
                              110, 114, 118]
        if train == 'train':
            self.image_list = self.get_list(training_set)
        elif train == 'val':
            self.image_list = self.get_list(validation_set)
        elif train == 'test':
            self.image_list = self.get_list(evaluation_set)

    def get_list(self, data_list):
        image_list = []
        pair_file = os.path.join(self.root_dir, 'proj_mat/img_pair.txt')
        image_pair = np.loadtxt(pair_file)
        image_pair = image_pair.reshape(-1, 2).astype(int)
        for f in data_list:
            dir = f'scan{f}'
            base_dir = os.path.join(self.root_dir, 'Rectified', dir)
            if os.path.exists(base_dir):
                for i, j in image_pair:
                    left_dir = os.path.join(base_dir, f'rect_0{i:02d}_1_r5000.png')
                    right_dir = os.path.join(base_dir, f'rect_0{j:02d}_1_r5000.png')
                    file_name = f'pos_0{i:02d}.txt'
                    left_poj_dir = os.path.join(self.root_dir, 'proj_mat', file_name)
                    file_name = f'pos_0{j:02d}.txt'
                    right_poj_dir = os.path.join(self.root_dir, 'proj_mat', file_name)
                    file_name = f'depth_map_00{i - 1:02d}.pfm'
                    depth = os.path.join(self.root_dir, 'Depths_raw', f'scan{f}', file_name)

                    image_list.append([left_dir, right_dir, left_poj_dir, right_poj_dir, depth])

        print(f"{len(image_list)} data collected")
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        left_dir, right_dir, left_poj_dir, right_poj_dir, depth = self.image_list[idx]
        left_image = img2tensor(left_dir)
        right_image = img2tensor(right_dir)
        image = torch.cat((left_image, right_image), dim=0)
        disparity = readPFM(depth)
        disparity = resize(disparity, (288, 384), anti_aliasing=False, order=0)
        disparity = torch.tensor(disparity, dtype=torch.float32).unsqueeze(0).float()

        if self.output_homo:
            left_poj = np.loadtxt(left_poj_dir).reshape(3, 4)
            right_poj = np.loadtxt(right_poj_dir).reshape(3, 4)
            right_poj_pinv = np.linalg.pinv(right_poj)

            # sx, sy = 384/(1600-300), 288/(1200-200)
            # S = np.diag([96 / 325, 0.288, 1])
            S = np.diag([0.24, 0.24, 1])
            S_inv = np.linalg.pinv(S)
            homo = S @ left_poj @ right_poj_pinv @ S_inv
            homo = torch.tensor(homo, dtype=torch.float32)

            return image, homo, disparity
        else:
            return image, disparity

    def get_max_disp(self):
        max_disp = -1
        for idx in range(len(self)):
            _, _, disp = self.__getitem__(idx)
            if torch.max(disp) > max_disp:
                max_disp = torch.max(disp)
        return max_disp


class ADT(Dataset):
    def __init__(self, root_dir, train=True, initial_list=True, sep_out=False):
        self.root_dir = root_dir
        self.train = train
        self.sep_out = sep_out
        if initial_list:
            file_list = []
            if train:
                txt = os.path.join(root_dir, 'train_list.txt')
            else:
                txt = os.path.join(root_dir, 'test_list.txt')
            with open(txt, 'r') as file:
                for line in file:
                    file_list.append(line.strip())
            self.image_list = self.get_file_list(file_list)
            print(f"Number of images: {len(self.image_list)}")

    def get_file_list(self, file_list):
        image_list = []
        if self.train:
            file = 'train'
        else:
            file = 'test'

        for level1 in file_list:
            path1 = os.path.join(self.root_dir, file, level1)
            for level2 in os.listdir(path1):
                path2 = os.path.join(path1, level2)
                for level3 in os.listdir(path2):
                    image_list.append(os.path.join(path2, level3))

        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx]
        left_img = iio.imread(os.path.join(path, 'left.png'))
        right_img = iio.imread(os.path.join(path, 'right.png'))
        depth = torch.load(os.path.join(path, 'depth.pt'))

        left_img = torch.from_numpy(left_img).permute(2, 0, 1).to(torch.float32)
        right_img = torch.from_numpy(right_img).repeat(3, 1, 1).to(torch.float32)

        if self.sep_out:
            return left_img, right_img, depth
        else:
            image = torch.cat((left_img, right_img), dim=0)
            return image, depth




