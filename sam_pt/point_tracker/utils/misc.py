# Adapted from:
#   - https://github.com/aharley/pips/blob/486124b4236bb228a20750b496f0fa8aa6343157/utils/misc.py
#   - https://github.com/aharley/pips2/blob/06bff81f25f2866728ff94f5d3a02c00893a8f15/utils/misc.py


import numpy as np
import torch


def posemb_sincos_2d_xy(xy, C, temperature=10000, cat_coords=False):
    device = xy.device
    dtype = xy.dtype
    B, S, D = xy.shape
    assert (D == 2)
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    assert (C % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(C // 4, device=device) / (C // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    pe = pe.reshape(B, S, C).type(dtype)
    if cat_coords:
        pe = torch.cat([pe, xy], dim=2)  # B,N,C+2
    return pe


def get_3d_embedding(xyz, C, cat_coords=True):
    B, N, D = xyz.shape
    assert (D == 3)

    x = xyz[:, :, 0:1]
    y = xyz[:, :, 1:2]
    z = xyz[:, :, 2:3]
    div_term = (torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)

    pe = torch.cat([pe_x, pe_y, pe_z], dim=2)  # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2)  # B, N, C*3+3
    return pe


class SimplePool():
    def __init__(self, pool_size, version='pt'):
        self.pool_size = pool_size
        self.version = version
        # random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.items = []
        if not (version == 'pt' or version == 'np'):
            print('version = %s; please choose pt or np')
            assert (False)  # please choose pt or np

    def __len__(self):
        return len(self.items)

    def mean(self, min_size='none'):
        if min_size == 'half':
            pool_size_thresh = self.pool_size / 2
        else:
            pool_size_thresh = 1

        if self.version == 'np':
            if len(self.items) >= pool_size_thresh:
                return np.sum(self.items) / float(len(self.items))
            else:
                return np.nan
        if self.version == 'pt':
            if len(self.items) >= pool_size_thresh:
                return torch.sum(self.items) / float(len(self.items))
            else:
                return torch.from_numpy(np.nan)

    def sample(self):
        idx = np.random.randint(len(self.items))
        return self.items[idx]

    def fetch(self, num=None):
        if self.version == 'pt':
            item_array = torch.stack(self.items)
        elif self.version == 'np':
            item_array = np.stack(self.items)
        if num is not None:
            # there better be some items
            assert (len(self.items) >= num)

            # if there are not that many elements just return however many there are
            if len(self.items) < num:
                return item_array
            else:
                idxs = np.random.randint(len(self.items), size=num)
                return item_array[idxs]
        else:
            return item_array

    def is_full(self):
        full = self.num == self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full

    def empty(self):
        self.items = []
        self.num = 0

    def update(self, items):
        for item in items:
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.items.pop(0)
            # add to the back
            self.items.append(item)
        return self.items
