import math
import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np


def w1(z):
    z1, z2 = z[:, 0], z[:, 1]
    return torch.sin(0.5 * math.pi * z1)


def helper(z, a):
    return torch.exp(-0.5 * ((z / a) ** 2))


def w2(z):
    z1 = z[:, 0]
    return 3 * helper(z1 - 1, 0.6)


def w3(z):
    z1 = z[:, 0]

    def sigma(z):
        return 1 / (1 + torch.exp(-z))

    return 3 * sigma((z1 - 1) / 0.3)


def cal_u1(z):
    z1 = z[:, 0]  # 第一维表示的是batch里面的编号
    res = 0
    res += 0.5 * ((torch.norm(z, dim=1) - 2) / 0.4) ** 2
    res -= torch.log(helper(z1 - 2, 0.6) + helper(z1 + 2, 0.6))
    res = torch.exp(-res)
    return res


def cal_u2(z):
    z1, z2 = z[:, 0], z[:, 1]
    res = 0.5 * ((z2 - w1(z)) / 0.4) ** 2
    return torch.exp(-res)


def cal_u3(z):
    z1, z2 = z[:, 0], z[:, 1]
    res = 0
    res += helper(z2 - w1(z), 0.35)
    res += helper(z2 - w1(z) + w2(z), 0.35)
    res = - torch.log(res)
    return torch.exp(-res)


def cal_u4(z):
    z1, z2 = z[:, 0], z[:, 1]
    res = 0
    res += helper(z2 - w1(z), 0.4)
    res += helper(z2 - w1(z) + w3(z), 0.35)
    res = - torch.log(res)
    return torch.exp(-res)


def sample_(base_x, base_u, device, batch_size):  # 用作四种平面曲线的采样函数
    ind_list = list(WeightedRandomSampler(base_u, num_samples=batch_size,  # 处理样本不均衡的采样，采用有放回模式
                                          replacement=True))  # 得到的ind_list是index

    z_t1 = base_x[:, ind_list]
    z_t1[1, :] = -base_x[1, ind_list]  # 翻转，不然不知道为啥图像不对
    z_t1 = torch.from_numpy(z_t1)
    z_t1 = z_t1.float()
    z_t1 = z_t1.to(device)
    z_t1 = z_t1.t()  # 一行为一点的坐标
    return z_t1


def helper_0():
    x1, x2 = np.arange(-4, 4, 0.005), np.arange(-4, 4, 0.005)  # 1600*1600
    x1, x2 = np.meshgrid(x1, x2)  # 横坐标矩阵与纵坐标矩阵
    _x1 = x1.ravel('C')  # 拉直，把一个meshgrid转为一个向量,以行序优先,为行向量
    _x2 = x2.ravel('C')
    x = np.vstack((_x1, _x2))  # 拼接，每一列对应于meshgrid上的一个点的坐标
    y = torch.from_numpy(x)
    y = y.t()  # 每一行对应于一个点的坐标
    return x, y  # x为numpy数组，y为tensor数组
