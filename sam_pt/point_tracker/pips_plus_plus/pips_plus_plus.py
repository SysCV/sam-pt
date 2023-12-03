# Taken from: https://github.com/aharley/pips2/blob/06bff81f25f2866728ff94f5d3a02c00893a8f15/nets/pips2.py

import torch
import torch.nn.functional as F
from torch import nn

import sam_pt.point_tracker.utils.basic
import sam_pt.point_tracker.utils.misc
import sam_pt.point_tracker.utils.samp


class Conv1dPad(nn.Module):
    """
    nn.Conv1d with auto-computed padding ("same" padding)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(Conv1dPad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, use_norm, use_do, is_first_block=False):
        super(ResidualBlock1d, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.stride = 1
        self.is_first_block = is_first_block
        self.use_norm = use_norm
        self.use_do = use_do

        self.norm1 = nn.InstanceNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = Conv1dPad(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups)

        self.norm2 = nn.InstanceNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = Conv1dPad(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups)

    def forward(self, x):

        identity = x

        out = x
        if not self.is_first_block:
            if self.use_norm:
                out = self.norm1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_norm:
            out = self.norm2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        out += identity
        return out


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    B, S, N, D = flow_gt.shape
    assert (D == 2)
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert (S == S1)
    assert (S == S2)
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]
        i_loss = (flow_pred - flow_gt).abs()  # B,S,N,2
        i_loss = torch.mean(i_loss, dim=3)  # B,S,N
        flow_loss += i_weight * sam_pt.point_tracker.utils.basic.reduce_masked_mean(i_loss, valids)
    flow_loss = flow_loss / n_predictions
    return flow_loss


class ResidualBlock2d(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode='zeros')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=8, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn

        self.in_planes = 64

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode='zeros')
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(128, stride=2)

        self.conv2 = nn.Conv2d(128 + 128 + 96 + 64, output_dim * 2, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock2d(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock2d(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        a = F.interpolate(a, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
        b = F.interpolate(b, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
        c = F.interpolate(c, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
        d = F.interpolate(d, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x


class DeltaBlock(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=128, corr_levels=4, corr_radius=3):
        super(DeltaBlock, self).__init__()

        kitchen_dim = (corr_levels * (2 * corr_radius + 1) ** 2) * 3 + latent_dim + 2

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        in_channels = kitchen_dim
        base_filters = 128
        self.n_block = 8
        self.kernel_size = 3
        self.groups = 1
        self.use_norm = True
        self.use_do = False

        self.increasefilter_gap = 2

        self.first_block_conv = Conv1dPad(in_channels=in_channels, out_channels=base_filters,
                                          kernel_size=self.kernel_size, stride=1)
        self.first_block_norm = nn.InstanceNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):

            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False

            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = ResidualBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                groups=self.groups,
                use_norm=self.use_norm,
                use_do=self.use_do,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        self.final_norm = nn.InstanceNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, 2)

    def forward(self, fcorr, flow):
        B, S, D = flow.shape
        assert (D == 2)
        flow_sincos = sam_pt.point_tracker.utils.misc.posemb_sincos_2d_xy(flow, self.latent_dim, cat_coords=True)
        x = torch.cat([fcorr, flow_sincos], dim=2)  # B,S,-1

        # conv1d wants channels in the middle
        out = x.permute(0, 2, 1)
        out = self.first_block_conv(out)
        out = self.first_block_relu(out)
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)
        out = self.final_relu(out)
        out = out.permute(0, 2, 1)

        delta = self.dense(out)
        return delta


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert (D == 2)

        x0 = coords[:, 0, :, 0].round().clamp(0, self.W - 1).long()
        y0 = coords[:, 0, :, 1].round().clamp(0, self.H - 1).long()

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B,S,N,H,W
            _, _, _, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(corrs.reshape(B * S * N, 1, H, W), coords_lvl)
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B,S,N,LRR*2
        return out.contiguous().float()

    def corr(self, targets):
        B, S, N, C = targets.shape
        assert (C == self.C)
        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)
            corrs = torch.matmul(targets, fmap2s)
            corrs = corrs.view(B, S, N, H, W)
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


class PipsPlusPlus(nn.Module):
    def __init__(self, stride=8):
        super(PipsPlusPlus, self).__init__()

        self.stride = stride

        self.hidden_dim = hdim = 256
        self.latent_dim = latent_dim = 128
        self.corr_levels = 4
        self.corr_radius = 3

        self.fnet = BasicEncoder(output_dim=self.latent_dim, norm_fn='instance', dropout=0, stride=stride)
        self.delta_block = DeltaBlock(hidden_dim=self.hidden_dim, corr_levels=self.corr_levels,
                                      corr_radius=self.corr_radius)
        self.norm = nn.GroupNorm(1, self.latent_dim)

    def forward(self, trajs_e0, rgbs, iters=3, trajs_g=None, vis_g=None, valids=None, sw=None, feat_init=None,
                is_train=False, beautify=False):
        total_loss = torch.tensor(0.0).cuda()

        B, S, N, D = trajs_e0.shape
        assert (D == 2)

        B, S, C, H, W = rgbs.shape
        rgbs = 2 * (rgbs / 255.0) - 1.0

        H8 = H // self.stride
        W8 = W // self.stride

        device = rgbs.device

        rgbs_ = rgbs.reshape(B * S, C, H, W)
        fmaps_ = self.fnet(rgbs_)
        fmaps = fmaps_.reshape(B, S, self.latent_dim, H8, W8)

        if sw is not None and sw.save_this:
            sw.summ_feats('1_model/0_fmaps', fmaps.unbind(1))

        coords = trajs_e0.clone() / float(self.stride)

        hdim = self.hidden_dim

        fcorr_fn1 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fn2 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fn4 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        if feat_init is not None:
            feats1, feats2, feats4 = feat_init
        else:
            feat1 = sam_pt.point_tracker.utils.samp.bilinear_sample2d(
                fmaps[:, 0], coords[:, 0, :, 0], coords[:, 0, :, 1]).permute(0, 2, 1)  # B,N,C
            feats1 = feat1.unsqueeze(1).repeat(1, S, 1, 1)  # B,S,N,C
            feats2 = feat1.unsqueeze(1).repeat(1, S, 1, 1)  # B,S,N,C
            feats4 = feat1.unsqueeze(1).repeat(1, S, 1, 1)  # B,S,N,C

        coords_bak = coords.clone()

        coords[:, 0] = coords_bak[:, 0]  # lock coord0 for target

        coord_predictions1 = []  # for loss
        coord_predictions2 = []  # for vis

        fcorr_fn1.corr(feats1)  # we only need to run this corr once

        coord_predictions2.append(coords.detach() * self.stride)

        for itr in range(iters):
            coords = coords.detach()

            if itr >= 1:
                # timestep indices
                inds2 = (torch.arange(S) - 2).clip(min=0)
                inds4 = (torch.arange(S) - 4).clip(min=0)
                # coordinates at these timesteps
                coords2_ = coords[:, inds2].reshape(B * S, N, 2)
                coords4_ = coords[:, inds4].reshape(B * S, N, 2)
                # featuremaps at these timesteps
                fmaps2_ = fmaps[:, inds2].reshape(B * S, self.latent_dim, H8, W8)
                fmaps4_ = fmaps[:, inds4].reshape(B * S, self.latent_dim, H8, W8)
                # features at these coords/times
                feats2_ = sam_pt.point_tracker.utils.samp.bilinear_sample2d(
                    fmaps2_, coords2_[:, :, 0], coords2_[:, :, 1]).permute(0, 2, 1)  # B*S, N, C
                feats2 = feats2_.reshape(B, S, N, self.latent_dim)
                feats4_ = sam_pt.point_tracker.utils.samp.bilinear_sample2d(
                    fmaps4_, coords4_[:, :, 0], coords4_[:, :, 1]).permute(0, 2, 1)  # B*S, N, C
                feats4 = feats4_.reshape(B, S, N, self.latent_dim)

            fcorr_fn2.corr(feats2)
            fcorr_fn4.corr(feats4)

            # now we want costs at the current locations
            fcorrs1 = fcorr_fn1.sample(coords)  # B,S,N,LRR
            fcorrs2 = fcorr_fn2.sample(coords)  # B,S,N,LRR
            fcorrs4 = fcorr_fn4.sample(coords)  # B,S,N,LRR
            LRR = fcorrs1.shape[3]

            # we want everything in the format B*N, S, C
            fcorrs1_ = fcorrs1.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            fcorrs2_ = fcorrs2.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            fcorrs4_ = fcorrs4.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            fcorrs_ = torch.cat([fcorrs1_, fcorrs2_, fcorrs4_], dim=2)
            flows_ = (coords[:, 1:] - coords[:, :-1]).permute(0, 2, 1, 3).reshape(B * N, S - 1, 2)
            flows_ = torch.cat([flows_, flows_[:, -1:]], dim=1)  # B*N,S,2

            delta_coords_ = self.delta_block(fcorrs_, flows_)  # B*N,S,2

            if beautify and itr > 3 * iters // 4:
                # this smooths the results a bit, but does not really help perf
                delta_coords_ = delta_coords_ * 0.5

            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)

            coord_predictions1.append(coords * self.stride)

            coords[:, 0] = coords_bak[:, 0]  # lock coord0 for target
            coord_predictions2.append(coords * self.stride)

        # pause at the end, to make the summs more interpretable
        coord_predictions2.append(coords * self.stride)

        if trajs_g is not None:
            loss = sequence_loss(coord_predictions1, trajs_g, vis_g, valids, 0.8)
        else:
            loss = None

        coord_predictions1.append(coords * self.stride)
        feats = (feats1, feats2, feats4)
        return coord_predictions1, coord_predictions2, feats, loss
