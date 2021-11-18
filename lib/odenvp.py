import torch
import torch.nn as nn
import lib.layers as layers
from lib.layers.odefunc import AutoencoderDiffEqNet, ODEnet
import numpy as np


class ODENVP(nn.Module):
    """
    Real NVP for image data. Will downsample the input until one of the
    dimensions is less than or equal to 4.

    Args:
        input_size (tuple): 4D tuple of the input size.
        n_scale (int): Number of scales for the representation z.
        n_resblocks (int): Length of the resnet for each coupling layer.
    """

    def __init__(
            self,
            input_size,
            n_scale=float('inf'),
            n_blocks=2,
            intermediate_dims=(32,),
            nonlinearity="softplus",
            squash_input=True,
            alpha=0.05,
            cnf_kwargs=None,
    ):
        super(ODENVP, self).__init__()
        self.n_scale = min(n_scale, self._calc_n_scale(input_size))  # 控制多尺度结构个数
        self.n_blocks = n_blocks  # 每个尺度结构前后堆叠CNF的个数
        self.intermediate_dims = intermediate_dims
        self.nonlinearity = nonlinearity
        self.squash_input = squash_input
        self.alpha = alpha
        self.cnf_kwargs = cnf_kwargs if cnf_kwargs else {}

        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]  # 元组元素为(c,h,w)

    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            transforms.append(
                StackedCNFLayers(
                    initial_size=(c, h, w),
                    idims=self.intermediate_dims,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=(layers.LogitTransform(self.alpha) if self.alpha > 0 else layers.ZeroMeanTransform())
                    if self.squash_input and i == 0 else None,
                    n_blocks=self.n_blocks,
                    cnf_kwargs=self.cnf_kwargs,
                    nonlinearity=self.nonlinearity,
                )
            )
            c, h, w = c * 2, h // 2, w // 2
        return nn.ModuleList(transforms)

    def get_regularization(self):
        if len(self.regularization_fns) == 0:
            return None

        acc_reg_states = tuple([0.] * len(self.regularization_fns))
        for module in self.modules():
            if isinstance(module, layers.CNF):
                acc_reg_states = tuple(
                    acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states())
                )
        return sum(state * coeff for state, coeff in zip(acc_reg_states, self.regularization_coeffs))

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2  # 将2调整为4
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def _logdensity(self, x, logpx=None, integration_times=None, update_t=False):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        out = []
        out_diff = []
        for idx in range(len(self.transforms)):
            if not update_t:
                x, _logpx = self.transforms[idx](x, _logpx, integration_times=integration_times, update_t=update_t)
            else:
                x, _logpx, z_diff, logpz_diff = self.transforms[idx](x, _logpx, integration_times=integration_times,
                                                                     update_t=update_t)
#            x, _logpx = self.transforms[idx](x, _logpx, integration_times=integration_times, update_t=update_t)
            if not update_t:
                if idx < len(self.transforms) - 1:
                    d = x.size(1) // 2
                    x, factor_out = x[:, :d], x[:, d:]  # 通道层面进行拆分
                else:
                    # last layer, no factor out
                    factor_out = x
                out.append(factor_out)
            else:
                if idx < len(self.transforms) - 1:
                    d = x.size(1) // 2
                    d_diff = z_diff.size(1) // 2
                    x, factor_out = x[:, :d], x[:, d:]  # 通道层面进行拆分
                    z_diff, factor_out_diff = z_diff[:, :d_diff], z_diff[:, d_diff:]
                else:
                    # last layer, no factor out
                    factor_out = x
                    factor_out_diff = z_diff
                out.append(factor_out)
                out_diff.append(factor_out_diff)
        if not update_t:
            out = [o.view(o.size()[0], -1) for o in out]
            out = torch.cat(out, 1)
        else:
            out = [o.view(o.size()[0], -1) for o in out]
            out_diff = [o.view(o.size()[0], -1) for o in out_diff]
            out = torch.cat(out, 1)
            out_diff = torch.cat(out_diff, 1)
        if not update_t:
            return out if logpx is None else (out, _logpx)
        else:
            return (out, out_diff, logpz_diff) if logpx is None else (out, _logpx, out_diff, logpz_diff)

    #        return (out, z_diff, logpz_diff) if logpx is None else (out, _logpx, z_diff, logpz_diff)

    def _generate(self, z, logpz=None, integration_times=None, update_t=False):
        z = z.view(z.shape[0], -1)
        zs = []
        i = 0
        for dims in self.dims:
            s = np.prod(dims)
            zs.append(z[:, i:i + s])
            i += s
        zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]
        _logpz = torch.zeros(zs[0].shape[0], 1).to(zs[0]) if logpz is None else logpz
        if not update_t:
            z_prev, _logpz = self.transforms[-1](zs[-1], _logpz, reverse=True, integration_times=integration_times,
                                                 update_t=update_t)
        else:
            z_prev, _logpz, z_diff, logpz_diff = self.transforms[-1](zs[-1], _logpz, reverse=True,
                                                                     integration_times=integration_times,
                                                                     update_t=update_t)
        #        z_prev, _logpz = self.transforms[-1](zs[-1], _logpz, reverse=True, integration_times=integration_times,
        #                                             update_t=update_t)
        for idx in range(len(self.transforms) - 2, -1, -1):
            z_prev = torch.cat((z_prev, zs[idx]), dim=1)
            if not update_t:
                z_prev, _logpz = self.transforms[idx](z_prev, _logpz, reverse=True, integration_times=integration_times,
                                                      update_t=update_t)
            else:
                z_prev, _logpz, z_diff, logpz_diff = self.transforms[idx](z_prev, _logpz, reverse=True,
                                                                          integration_times=integration_times,
                                                                          update_t=update_t)
#            z_prev, _logpz = self.transforms[idx](z_prev, _logpz, reverse=True, integration_times=integration_times,
        #                                                  update_t=update_t)
        if not update_t:
            return z_prev if logpz is None else (z_prev, _logpz)
        else:
            return (z_prev, z_diff, logpz_diff) if logpz is None else (z_prev, _logpz, z_diff, logpz_diff)

    #        return (z_prev, z_diff, logpz_diff) if logpz is None else (z_prev, _logpz, z_diff, logpz_diff)

    def forward(self, x, logpx=None, reverse=False, integration_times=None, update_t=False):
        if reverse:
            return self._generate(x, logpx, integration_times=integration_times, update_t=update_t)
        else:
            return self._logdensity(x, logpx, integration_times=integration_times, update_t=update_t)


class StackedCNFLayers(layers.SequentialFlow):
    def __init__(
            self,
            initial_size,
            idims=(32,),
            nonlinearity="softplus",
            squeeze=True,
            init_layer=None,
            n_blocks=1,
            cnf_kwargs={},
    ):
        strides = tuple([1] + [1 for _ in idims])
        chain = []
        if init_layer is not None:
            chain.append(init_layer)

        def _make_odefunc(size):
            net = AutoencoderDiffEqNet(idims, size, strides, True, layer_type="concat", nonlinearity=nonlinearity)
            f = layers.AutoencoderODEfunc(net)
            return f

        if squeeze:
            c, h, w = initial_size
            after_squeeze_size = c * 4, h // 2, w // 2
            pre = [layers.CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]
            post = [layers.CNF(_make_odefunc(after_squeeze_size), **cnf_kwargs) for _ in range(n_blocks)]
            chain += pre + [layers.SqueezeLayer(2)] + post
        else:
            chain += [layers.CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]

        super(StackedCNFLayers, self).__init__(chain)
