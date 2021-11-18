import torch.nn as nn
import lib.layers as layers


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reverse=False, inds=None, integration_times=None, update_t=False):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)  # 从后往前排列
            else:
                inds = range(len(self.chain))

        if not update_t:
            if logpx is None:
                for i in inds:
                    x = self.chain[i](x, reverse=reverse, integration_times=integration_times)
                return x
            else:
                for i in inds:
                    x, logpx = self.chain[i](x, logpx, reverse=reverse, integration_times=integration_times)
                return x, logpx
        else:
            if logpx is None:
                for i in inds:
                    if isinstance(self.chain[i], layers.CNF):
                        x, z_diff, logpz_diff = self.chain[i](x, reverse=reverse, integration_times=integration_times,
                                                              update_t=update_t)
                    else:
                        x = self.chain[i](x, reverse=reverse, integration_times=integration_times)
                return x, z_diff, logpz_diff
            else:
                for i in inds:
                    if isinstance(self.chain[i], layers.CNF):
                        x, logpx, z_diff, logpz_diff = self.chain[i](x, logpx, reverse=reverse,
                                                                     integration_times=integration_times,
                                                                     update_t=update_t)
                    else:
                        x, logpx = self.chain[i](x, logpx, reverse=reverse)

                return x, logpx, z_diff, logpz_diff
