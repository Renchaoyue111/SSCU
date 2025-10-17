from __future__ import absolute_import

import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def sample_cam(dataset):
    """
    for train sets, randomly assign samples to different devices
    :param dataset:
    :return:
    """
    dict_users, all_idxs, all_cams = {}, np.arange(len(dataset)), np.asarray([val[2] for val in dataset])
    cam_num = len(set(all_cams))
    if min(all_cams) > 0:
        all_cams -= 1
    for i in range(cam_num):
        dict_users[i] = all_idxs[i == all_cams].tolist()
    return dict_users, cam_num  # indexs and number of cams


def sample_id(dataset, num_users):
    """
    for train sets, randomly assign samples to different devices
    :param dataset:
    :return:
    """
    dict_users, all_idxs, all_ids = {}, np.arange(len(dataset)), np.asarray([val[1] for val in dataset])
    id_num = len(set(all_ids))
    per_num = id_num // num_users
    for i in range(num_users):
        dict_users[i] = all_idxs[(per_num * i <= all_ids) * (all_ids < per_num * (1 + i))].tolist()
    return dict_users


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


def init_zero(m):
    try:
        nn.init.constant_(m.weight, 0)
    except:
        pass
    try:
        nn.init.constant_(m.bias, 0)
    except:
        pass
    try:
        nn.init.constant_(m.running_mean, 0)
    except:
        pass
    try:
        nn.init.constant_(m.running_var, 0)
    except:
        pass


def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m
