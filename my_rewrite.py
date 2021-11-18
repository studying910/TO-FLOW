#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib

matplotlib.use('Agg')  # 不显示绘图,用来生成图像文件
import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint_adjoint as odeint



# In[ ]:
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def divergence_approx(f, z, e=None):  # f为z的函数，z为自变量，e为随机噪声列向量
    """
    TORCH.AUTOGRAD.GRAD:Computes and returns the sum of gradients of outputs with respect to the inputs.

    torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None,
                        create_graph=False, only_inputs=True, allow_unused=False)
       grad_outputs should be a sequence of length matching output containing the “vector” in Jacobian-vector product.
       乘以一个向量是为了方便得到\partial f / \partial z 中z的每一维的导数

    f : [batchsize, D]
    z : [batchsize, D]
    e : [batchsize, D]
    Parameters
    ----------
    f : TYPE
        \partial f / \partial z 中的f.
    z : TYPE
        \partial f / \partial z 中的z.
    e : Tensor, optional
        Hutchinson估计中的随机数，服从标准正态或者Rademacher分布. The default is None.

    Returns
    -------
    approx_tr_dzdx : Tensor     Size : [batch_size, ]
        迹的Hutchinson估计.

    """
    e_dfdz = torch.autograd.grad(f, z, e, create_graph=True)[0]  # 最后的这个[0]一定得有，不然下一行那个乘法运算没法进行
    e_dfdz_e = e_dfdz * e  # 这里e_dfdz与e的size是一样的，它们相乘是对应元素相乘
    approx_tr_dfdz = e_dfdz_e.view(z.shape[0], -1).sum(dim=1)  # 对行求和
    return approx_tr_dfdz  # D行1列的向量


def sample_rademacher_like(y):
    """
    生成服从Rademacher分布的随机数，randint生成的随机数的取值范围为[low,high)

    """
    # 这里的.to(y)是为了保证生成的rademacher随机数的datatype以及device与y相同
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


class myodefunc(nn.Module):
    """
    返回eq.4中的被积函数，其中:
        z_dot : f(z(t), t;theta)
        -divergence.view(batchsize, 1) : -Tr
    """

    def __init__(self, Z_DIM, hidden_dims=(64, 64), activation="relu"):
        super(myodefunc, self).__init__()
        # Define net
        self.activation = activation
        self.layers = []
        self._e = None  # Hutchison估计里面的随机噪声
        dim_list = [Z_DIM] + list(hidden_dims) + [Z_DIM]
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i] + 1, dim_list[i + 1]))
        self.layers = nn.ModuleList(layers)

    def get_z_dot(self, t, z):
        """z_dot is parameterized by a NN: z_dot = f(t, z(t))"""
        """这里我有点疑问，把t和z合并为一个向量，在得到结果之后为什么不把t提取出来，使z和z_dot的维度保持一致"""
        z_dot = z
        for l, layer in enumerate(self.layers):
            # Concatenate t at each layer.
            tz_cat = torch.cat((t.expand(z.shape[0], 1), z_dot), dim=1)  # D行3列的向量
            z_dot = layer(tz_cat)
            if l < len(self.layers) - 1:
                if self.activation == "relu":
                    z_dot = F.relu(z_dot)
                elif self.activation == "softplus":
                    z_dot = F.softplus(z_dot)
        return z_dot

    def forward(self, t, states):
        """
        Calculate the time derivative of z and divergence.
        Parameters
        ----------
        t : torch.Tensor
        time
        states : tuple
        Contains two torch.Tensors: z and delta_logpz
        Returns
        -------
        z_dot : torch.Tensor
        Time derivative of z.
        negative_divergence : torch.Tensor
        Time derivative of the log Jacobian.
        """
        z = states[0]
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):  # 可以求导
            z.requires_grad_(True)
            t.requires_grad_(True)

            if self._e is None:
                self._e = sample_rademacher_like(z)
            # Calculate the time derivative of z.
            # This is f(z(t), t; \theta) in Eq. 4.
            z_dot = self.get_z_dot(t, z)  # D行3列向量，第一列是t变换得到，第二三列是z变换得到

            # Calculate the time derivative of the log Jacobian.
            # This is -Tr(\partial z_dot / \partial z(t)) in Eq.s 2-4.
            #
            # Note that this is the brute force, O(D^2), method. This is fine
            # for D=2, but the authors suggest using a Monte-carlo estimate
            # of the trace (Hutchinson's trace estimator, eq. 7) for a linear
            # time estimate in larger dimensions.

            # divergence = 0.0
            #
            # for i in range(z.shape[1]):
            #     divergence += torch.autograd.grad(z_dot[:, i].sum(), z, create_graph=True, retain_graph=True)[0][:, i]
            if self.training:  # 处于训练模式下
                # 计算迹的Hutchinson估计
                divergence = divergence_approx(z_dot, z, self._e).view(batchsize, 1)
            else:
                # 强行计算秩
                divergence = 0.0
                for i in range(z.shape[1]):  # 遍历雅克比矩阵的对角线元素并求和
                    divergence += torch.autograd.grad(z_dot[:, i].sum(),
                                                      z, create_graph=True, retain_graph=True)[0][:, i]
            torch.cuda.empty_cache()

        return z_dot, -divergence.view(batchsize, 1)


# In[ ]:
class cnf_net(torch.nn.Module):
    """Continuous noramlizing flow model."""

    def __init__(self, dim, device='cuda', func=None, activation="relu"):
        super(cnf_net, self).__init__()
        self.activation = activation
        if func is None:
            odefunc = myodefunc(dim, activation=self.activation).to(device)
        self.time_deriv_func = odefunc  # 可以得到eq.4中被积函数的-Tr项
        self.device = torch.device("cuda" if (device == "cuda") & torch.cuda.is_available() else "cpu")
        self.integration_times = 10
        self.activation = activation

    def save_state(self, fn='state.tar'):
        """Save model state."""
        torch.save(self.state_dict(), fn)

    def load_state(self, fn='state.tar'):
        """Load model state."""
        self.load_state_dict(torch.load(fn, map_location='cpu'))

    def forward(self, z, delta_logpz=None, integration_times=None, reverse=False):
        """
        Implementation of Eq. 4.
        We want to integrate both f and the trace term. During training, we
        integrate from t_1 (data distribution) to t_0 (base distibution).
        Parameters
        ----------
        z : torch.Tensor
            Samples.
        delta_logpz : torch.Tensor
            Log determininant of the Jacobian.
        integration_times : torch.Tensor
            Which times to evaluate at.
        reverse : bool, optional
            Whether to reverse the integration times.
        Returns
        -------
        z : torch.Tensor
            Updated samples.
        delta_logpz : torch.Tensor
            Updated log determinant term.
        """
        device = self.device
        if delta_logpz is None:
            delta_logpz = torch.zeros(z.shape[0], 1).to(device)
        if integration_times is None:
            integration_times = torch.tensor([0.0, 10.0]).to(z)  # 积分计算的时间点
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Integrate. This is the call to torchdiffeq.
        state = odeint(
            self.time_deriv_func,  # Calculates time derivatives.
            (z, delta_logpz),  # Values to update. 相当于初值
            integration_times,  # When to evaluate.
            method='dopri5',  # Runge-Kutta
            atol=1e-4,  # Absolute tolerance
            rtol=1e-4,  # Relative tolerance
        )

        if len(integration_times) == 2:
            state = tuple(s[1] for s in state)
        z, delta_logpz = state
        return z, delta_logpz


import numpy as np


def change(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    return x


class Adam_t():  # 用于时间t更新的优化器
    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = None
        self.v = None
        self.n = 0

    def __call__(self, params, grads):  # 可通过'实例名()'来调用此方法
        params = change(params)
        grads = change(grads)
        if self.m is None:
            self.m = np.zeros_like(params)
        if self.v is None:
            self.v = np.zeros_like(params)

        self.n += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))  # 自适应调整学习率
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        # print(params)
        # print(alpha)
        params -= alpha * self.m / (np.sqrt(self.v) + self.eps)

        return params
