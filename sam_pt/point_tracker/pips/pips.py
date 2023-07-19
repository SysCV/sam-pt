# Taken from: https://github.com/aharley/pips/blob/486124b4236bb228a20750b496f0fa8aa6343157/nets/pips.py

from functools import partial

import torch
import torch.nn.functional as F
from einops.layers.torch import Reduce
from torch import nn

import sam_pt.point_tracker.utils.basic
import sam_pt.point_tracker.utils.misc
import sam_pt.point_tracker.utils.samp


def balanced_ce_loss(pred, gt, valid=None):
    # pred and gt are the same shape
    for (a, b) in zip(pred.size(), gt.size()):
        assert (a == b)  # some shape mismatch!
    if valid is not None:
        for (a, b) in zip(pred.size(), valid.size()):
            assert (a == b)  # some shape mismatch!
    else:
        valid = torch.ones_like(gt)

    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()

    label = pos * 2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

    pos_loss = sam_pt.point_tracker.utils.basic.reduce_masked_mean(loss, pos * valid)
    neg_loss = sam_pt.point_tracker.utils.basic.reduce_masked_mean(loss, neg * valid)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss, loss


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
        flow_pred = flow_preds[i]  # [:,:,0:1]
        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = torch.mean(i_loss, dim=3)  # B, S, N
        flow_loss += i_weight * sam_pt.point_tracker.utils.basic.reduce_masked_mean(i_loss, valids)
    flow_loss = flow_loss / n_predictions
    return flow_loss


def score_map_loss(fcps, trajs_g, vis_g, valids):
    #     # fcps is B,S,I,N,H8,W8
    B, S, I, N, H8, W8 = fcps.shape
    fcp_ = fcps.permute(0, 1, 3, 2, 4, 5).reshape(B * S * N, I, H8, W8)  # BSN,I,H8,W8
    # print('fcp_', fcp_.shape)
    xy_ = trajs_g.reshape(B * S * N, 2).round().long()  # BSN,2
    vis_ = vis_g.reshape(B * S * N)  # BSN
    valid_ = valids.reshape(B * S * N)  # BSN
    x_, y_ = xy_[:, 0], xy_[:, 1]  # BSN
    ind = (x_ >= 0) & (x_ <= (W8 - 1)) & (y_ >= 0) & (y_ <= (H8 - 1)) & (valid_ > 0) & (vis_ > 0)  # BSN
    fcp_ = fcp_[ind]  # N_,I,H8,W8
    xy_ = xy_[ind]  # N_,2
    N_ = fcp_.shape[0]
    # N_ is the number of heatmaps with valid targets

    # make gt with ones at the rounded spatial inds in here
    gt_ = torch.zeros_like(fcp_)  # N_,I,H8,W8
    for n in range(N_):
        gt_[n, :, xy_[n, 1], xy_[n, 0]] = 1  # put a 1 in the right spot, for each el in I

    ## softmax
    # fcp_ = fcp_.reshape(N_*I,H8*W8)
    # gt_ = gt_.reshape(N_*I,H8*W8)
    # argm = torch.argmax(gt_, dim=1)
    # ce_loss = F.cross_entropy(fcp_, argm, reduction='mean')

    ## ce
    fcp_ = fcp_.reshape(N_ * I * H8 * W8)
    gt_ = gt_.reshape(N_ * I * H8 * W8)
    # ce_loss = F.binary_cross_entropy_with_logits(fcp_, gt_, reduction='mean')
    ce_loss, _ = balanced_ce_loss(fcp_, gt_)
    # print('ce_loss', ce_loss)
    return ce_loss


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(S, input_dim, dim, output_dim, depth, expansion_factor=4, dropout=0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        nn.Linear(input_dim, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(S, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, output_dim)
    )


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

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
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

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
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode='zeros')
        self.relu1 = nn.ReLU(inplace=True)

        self.shallow = False
        if self.shallow:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128 + 96 + 64, output_dim, kernel_size=1)
        else:
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
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(a, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            b = F.interpolate(b, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            c = F.interpolate(c, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat([a, b, c], dim=1))
        else:
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
    def __init__(self, input_dim=128, hidden_dim=128, corr_levels=4, corr_radius=3, S=8):
        super(DeltaBlock, self).__init__()

        self.input_dim = input_dim

        kitchen_dim = (corr_levels * (2 * corr_radius + 1) ** 2) + input_dim + 64 * 3 + 3

        self.hidden_dim = hidden_dim

        self.S = S

        self.to_delta = MLPMixer(
            S=self.S,
            input_dim=kitchen_dim,
            dim=512,
            output_dim=self.S * (input_dim + 2),
            depth=12,
        )

    def forward(self, fhid, fcorr, flow):
        B, S, D = flow.shape
        assert (D == 3)
        flow_sincos = sam_pt.point_tracker.utils.misc.get_3d_embedding(flow, 64, cat_coords=True)
        x = torch.cat([fhid, fcorr, flow_sincos], dim=2)  # B, S, -1
        delta = self.to_delta(x)
        delta = delta.reshape(B, self.S, self.input_dim + 2)
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
        # print('fmaps', fmaps.shape)
        self.S, self.C, self.H, self.W = S, C, H, W

        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        # print('fmaps', fmaps.shape)

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)
            # print('fmaps', fmaps.shape)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert (D == 2)

        x0 = coords[:, 0, :, 0].round().clamp(0, self.W - 1).long()
        y0 = coords[:, 0, :, 1].round().clamp(0, self.H - 1).long()

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
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

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        return out.contiguous().float()

    def corr(self, targets):
        B, S, N, C = targets.shape
        assert (C == self.C)
        assert (S == self.S)

        fmap1 = targets

        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


class Pips(nn.Module):
    def __init__(self, S=8, stride=8):
        super(Pips, self).__init__()

        self.S = S
        self.stride = stride

        self.hidden_dim = hdim = 256
        self.latent_dim = latent_dim = 128
        self.corr_levels = 4
        self.corr_radius = 3

        self.fnet = BasicEncoder(output_dim=self.latent_dim, norm_fn='instance', dropout=0, stride=stride)

        self.delta_block = DeltaBlock(input_dim=self.latent_dim, hidden_dim=self.hidden_dim,
                                      corr_levels=self.corr_levels, corr_radius=self.corr_radius, S=self.S)

        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            # nn.GroupNorm(1, self.latent_dim),
            # nn.Linear(self.latent_dim, self.latent_dim),
            # nn.GELU(),
            nn.Linear(self.latent_dim, 1),
        )

    def forward(self, xys, rgbs, coords_init=None, feat_init=None, iters=3, trajs_g=None, vis_g=None, valids=None,
                sw=None, return_feat=False, is_train=False):
        B, N, D = xys.shape
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

        xys_ = xys.clone() / float(self.stride)

        if coords_init is None:
            coords = xys_.reshape(B, 1, N, 2).repeat(1, S, 1, 1)  # init with zero vel
        else:
            coords = coords_init.clone() / self.stride

        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        if feat_init is None:
            # initialize features for the whole traj, using the initial feature
            ffeat = sam_pt.point_tracker.utils.samp.bilinear_sample2d(fmaps[:, 0], coords[:, 0, :, 0],
                                                                              coords[:, 0, :, 1]).permute(0, 2,
                                                                                                          1)  # B, N, C
        else:
            ffeat = feat_init
        ffeats = ffeat.unsqueeze(1).repeat(1, S, 1, 1)  # B, S, N, C

        coords_bak = coords.clone()

        coord_predictions = []
        coord_predictions2 = []

        # pause at beginning
        coord_predictions2.append(coords.detach() * self.stride)
        coord_predictions2.append(coords.detach() * self.stride)

        fcps = []
        kps = []

        if sw is not None and sw.save_this:
            kp_vis = []
            # vis init
            for s in range(S):
                if trajs_g is not None:
                    e_ = coords_bak[0:1, s, 0:1]  # 1,1,2, in H8,W8 coords
                    g_ = trajs_g[0:1, s, 0:1] / float(self.stride)  # 1,1,2, in H8,W8 coords
                    kp = pips_utils.improc.draw_circles_at_xy(torch.cat([e_, g_], dim=1), H8, W8, sigma=1).squeeze(2)
                    kp = pips_utils.improc.seq2color(kp, colormap='onediff')
                else:
                    kp = pips_utils.improc.draw_circles_at_xy(coords[0:1, s, 0:1], H8, W8, sigma=1).squeeze(2)
                    kp = pips_utils.improc.seq2color(kp, colormap='spring')
                kp = pips_utils.improc.back2color(kp)
                kp_vis.append(kp)
            kp_vis = torch.stack(kp_vis, dim=1)
            kps.append(kp_vis)

        for itr in range(iters):
            coords = coords.detach()

            fcorr_fn.corr(ffeats)

            fcp = torch.zeros((B, S, N, H8, W8), dtype=torch.float32, device=device)  # B,S,N,H8,W8
            for cr in range(self.corr_levels):
                fcp_ = fcorr_fn.corrs_pyramid[cr]  # B,S,N,?,? (depending on scale)
                _, _, _, H_, W_ = fcp_.shape
                fcp_ = fcp_.reshape(B * S, N, H_, W_)
                fcp_ = F.interpolate(fcp_, (H8, W8), mode='bilinear', align_corners=True)
                fcp = fcp + fcp_.reshape(B, S, N, H8, W8)
            fcps.append(fcp)

            fcorrs = fcorr_fn.sample(coords)  # B, S, N, LRR
            LRR = fcorrs.shape[3]

            # for mixer, i want everything in the format B*N, S, C
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)
            times_ = torch.linspace(0, S, S, device=device).reshape(1, S, 1).repeat(B * N, 1, 1)  # B*N,S,1
            flows_ = torch.cat([flows_, times_], dim=2)  # B*N,S,2

            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            delta_all_ = self.delta_block(ffeats_, fcorrs_, flows_)  # B*N, S, C+2
            delta_coords_ = delta_all_[:, :, :2]
            delta_feats_ = delta_all_[:, :, 2:]

            ffeats_ = ffeats_.reshape(B * N * S, self.latent_dim)
            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
            ffeats_ = self.ffeat_updater(self.norm(delta_feats_)) + ffeats_
            ffeats = ffeats_.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)  # B,S,N,C

            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)

            if not is_train:
                coords[:, 0] = coords_bak[:, 0]  # lock coord0 for target

            coord_predictions.append(coords * self.stride)
            coord_predictions2.append(coords * self.stride)

            if sw is not None and sw.save_this:
                kp_vis = []
                for s in range(S):

                    if trajs_g is not None:
                        e_ = coords[0:1, s, 0:1]  # 1,1,2, in H8,W8 coords
                        g_ = trajs_g[0:1, s, 0:1] / float(self.stride)  # 1,1,2, in H8,W8 coords
                        kp = pips_utils.improc.draw_circles_at_xy(torch.cat([e_, g_], dim=1), H8, W8, sigma=1).squeeze(
                            2)
                        kp = pips_utils.improc.seq2color(kp, colormap='onediff')
                    else:
                        kp = pips_utils.improc.draw_circles_at_xy(coords[0:1, s, 0:1], H8, W8, sigma=1).squeeze(2)
                        kp = pips_utils.improc.seq2color(kp, colormap='spring')

                    kp = pips_utils.improc.back2color(kp)
                    kp_vis.append(kp)
                kp_vis = torch.stack(kp_vis, dim=1)
                kps.append(kp_vis)

        vis_e = self.vis_predictor(ffeats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)

        # pause at the end
        coord_predictions2.append(coords * self.stride)
        coord_predictions2.append(coords * self.stride)

        fcps = torch.stack(fcps, dim=2)  # B, S, I, N, H8, W8
        if sw is not None and sw.save_this:
            kps = torch.stack(kps, dim=2)  # B, S, I, 3, H8, W8

            vis_all = []
            vis_fcp = []

            fcps_ = fcps[0:1, :, :, 0:1].detach()  # 1,S,I,N,H8,W8
            fcps_ = sam_pt.point_tracker.utils.basic.normalize(fcps_)
            for s in range(S):
                fcp = fcps_[0:1, s, :, 0:1]  # 1,I,1,H8,W8
                fcp = torch.cat([fcp[:, 0].unsqueeze(1),  # zeroth
                                 fcp,
                                 fcp[:, -1].unsqueeze(1),
                                 fcp[:, -1].unsqueeze(1)], dim=1)  # pause on end
                fcp_vis = sw.summ_oneds('1_model/2_fcp_s%d' % s, fcp.unbind(1), norm=False, only_return=True)
                vis_fcp.append(fcp_vis)

                kp = kps[0:1, s]  # 1, I, 3, H8, W8
                kp = torch.cat([kp,
                                kp[:, -1].unsqueeze(1),
                                kp[:, -1].unsqueeze(1)], dim=1)  # pause on end
                kp_vis = sw.summ_rgbs('1_model/2_kp_s%d' % s, kp.unbind(1), only_return=True)

                kp_any = (torch.max(kp_vis, dim=2, keepdims=True)[0]).repeat(1, 1, 3, 1, 1)
                kp_vis[kp_any == 0] = fcp_vis[kp_any == 0]

                vis_all.append(kp_vis)
            vis_all = torch.stack(vis_all, dim=1)  # B, S, I, 3, H8, W8 (but not quite I, due to padding)
            vis_fcp = torch.stack(vis_fcp, dim=1)  # B, S, I, 3, H8, W8 (but not quite I, due to padding)

            vis_all = vis_all.permute(0, 2, 3, 1, 4, 5).reshape(1, -1, 3, S * H8, W8)
            vis_fcp = vis_fcp.permute(0, 2, 3, 1, 4, 5).reshape(1, -1, 3, S * H8, W8)
            sw.summ_rgbs('1_model/2_kp_s', vis_all.unbind(1))

        if trajs_g is not None:
            seq_loss = sequence_loss(coord_predictions, trajs_g, vis_g, valids, 0.8)
            vis_loss, _ = balanced_ce_loss(vis_e, vis_g, valids)
            ce_loss = score_map_loss(fcps, trajs_g / float(self.stride), vis_g, valids)
            losses = (seq_loss, vis_loss, ce_loss)
        else:
            losses = None

        if return_feat:
            return coord_predictions, coord_predictions2, vis_e, ffeat, losses
        else:
            return coord_predictions, coord_predictions2, vis_e, losses
