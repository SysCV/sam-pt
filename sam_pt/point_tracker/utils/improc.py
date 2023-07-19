# Taken from: https://github.com/aharley/pips/blob/486124b4236bb228a20750b496f0fa8aa6343157/utils/improc.py

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from sklearn.decomposition import PCA

import sam_pt.point_tracker.utils.basic


def preprocess_color_tf(x):
    import tensorflow as tf
    return tf.cast(x, tf.float32) * 1. / 255 - 0.5


def preprocess_color(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) * 1. / 255 - 0.5
    else:
        return x.float() * 1. / 255 - 0.5


def pca_embed(emb, keep, valid=None, eps=1e-6):
    ## emb -- [S,H/2,W/2,C]
    ## keep is the number of principal components to keep
    ## Helper function for reduce_emb.
    emb = emb + eps
    # emb is B x C x H x W
    emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy()  # this is B x H x W x C

    if valid:
        _, H, W, _ = emb.shape
        valid = valid.cpu().detach().numpy().reshape((H * W))

    emb_reduced = list()

    B, H, W, C = np.shape(emb)
    for img in emb:
        if np.isnan(img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        pixels_kd = np.reshape(img, (H * W, C))

        if valid:
            pixels_kd_pca = pixels_kd[valid]
        else:
            pixels_kd_pca = pixels_kd

        P = PCA(keep)
        P.fit(pixels_kd_pca)

        if valid:
            pixels3d = P.transform(pixels_kd) * valid
        else:
            pixels3d = P.transform(pixels_kd)

        out_img = np.reshape(pixels3d, [H, W, keep]).astype(np.float32)
        if np.isnan(out_img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        emb_reduced.append(out_img)

    emb_reduced = np.stack(emb_reduced, axis=0).astype(np.float32)

    return torch.from_numpy(emb_reduced).permute(0, 3, 1, 2)


def pca_embed_together(emb, keep, eps=1e-6):
    ## emb -- [S,H/2,W/2,C]
    ## keep is the number of principal components to keep
    ## Helper function for reduce_emb.
    emb = emb + eps
    # emb is B x C x H x W
    emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy()  # this is B x H x W x C

    B, H, W, C = np.shape(emb)
    if np.isnan(emb).any():
        return torch.zeros(B, keep, H, W)

    pixelskd = np.reshape(emb, (B * H * W, C))
    P = PCA(keep)
    P.fit(pixelskd)
    pixels3d = P.transform(pixelskd)
    out_img = np.reshape(pixels3d, [B, H, W, keep]).astype(np.float32)

    if np.isnan(out_img).any():
        return torch.zeros(B, keep, H, W)

    return torch.from_numpy(out_img).permute(0, 3, 1, 2)


def reduce_emb(emb, valid=None, inbound=None, together=False):
    ## emb -- [S,C,H/2,W/2], inbound -- [S,1,H/2,W/2]
    ## Reduce number of chans to 3 with PCA. For vis.
    # S,H,W,C = emb.shape.as_list()
    S, C, H, W = list(emb.size())
    keep = 3

    if together:
        reduced_emb = pca_embed_together(emb, keep)
    else:
        reduced_emb = pca_embed(emb, keep, valid)  # not im

    reduced_emb = sam_pt.point_tracker.utils.basic.normalize(reduced_emb) - 0.5
    if inbound is not None:
        emb_inbound = emb * inbound
    else:
        emb_inbound = None

    return reduced_emb, emb_inbound


def get_feat_pca(feat, valid=None):
    B, C, D, W = list(feat.size())
    # feat is B x C x D x W. If 3D input, average it through Height dimension before passing into this function.

    pca, _ = reduce_emb(feat, valid=valid, inbound=None, together=True)
    # pca is B x 3 x W x D
    return pca


def gif_and_tile(ims, just_gif=False):
    S = len(ims)
    # each im is B x H x W x C
    # i want a gif in the left, and the tiled frames on the right
    # for the gif tool, this means making a B x S x H x W tensor
    # where the leftmost part is sequential and the rest is tiled
    gif = torch.stack(ims, dim=1)
    if just_gif:
        return gif
    til = torch.cat(ims, dim=2)
    til = til.unsqueeze(dim=1).repeat(1, S, 1, 1, 1)
    im = torch.cat([gif, til], dim=3)
    return im


def back2color(i, blacken_zeros=False):
    if blacken_zeros:
        const = torch.tensor([-0.5]).to(i.device)
        i = torch.where(i == 0.0, const, i)
        return back2color(i)
    else:
        return ((i + 0.5) * 255).type(torch.ByteTensor)


def xy2heatmap(xy, sigma, grid_xs, grid_ys, norm=False):
    # xy is B x N x 2, containing float x and y coordinates of N things
    # grid_xs and grid_ys are B x N x Y x X

    B, N, Y, X = list(grid_xs.shape)

    mu_x = xy[:, :, 0].clone()
    mu_y = xy[:, :, 1].clone()

    x_valid = (mu_x > -0.5) & (mu_x < float(X + 0.5))
    y_valid = (mu_y > -0.5) & (mu_y < float(Y + 0.5))
    not_valid = ~(x_valid & y_valid)

    mu_x[not_valid] = -10000
    mu_y[not_valid] = -10000

    mu_x = mu_x.reshape(B, N, 1, 1).repeat(1, 1, Y, X)
    mu_y = mu_y.reshape(B, N, 1, 1).repeat(1, 1, Y, X)

    sigma_sq = sigma * sigma
    # sigma_sq = (sigma*sigma).reshape(B, N, 1, 1)
    sq_diff_x = (grid_xs - mu_x) ** 2
    sq_diff_y = (grid_ys - mu_y) ** 2

    term1 = 1. / 2. * np.pi * sigma_sq
    term2 = torch.exp(-(sq_diff_x + sq_diff_y) / (2. * sigma_sq))
    gauss = term1 * term2

    if norm:
        # normalize so each gaussian peaks at 1
        gauss_ = gauss.reshape(B * N, Y, X)
        gauss_ = sam_pt.point_tracker.utils.basic.normalize(gauss_)
        gauss = gauss_.reshape(B, N, Y, X)

    return gauss


def xy2heatmaps(xy, Y, X, sigma=30.0):
    # xy is B x N x 2

    B, N, D = list(xy.shape)
    assert (D == 2)

    device = xy.device

    grid_y, grid_x = sam_pt.point_tracker.utils.basic.meshgrid2d(B, Y, X, device=device)
    # grid_x and grid_y are B x Y x X
    grid_xs = grid_x.unsqueeze(1).repeat(1, N, 1, 1)
    grid_ys = grid_y.unsqueeze(1).repeat(1, N, 1, 1)
    heat = xy2heatmap(xy, sigma, grid_xs, grid_ys, norm=True)
    return heat


def draw_circles_at_xy(xy, Y, X, sigma=12.5):
    B, N, D = list(xy.shape)
    assert (D == 2)
    prior = xy2heatmaps(xy, Y, X, sigma=sigma)
    # prior is B x N x Y x X
    prior = (prior > 0.5).float()
    return prior


def seq2color(im, norm=True, colormap='coolwarm'):
    B, S, H, W = list(im.shape)
    # S is sequential

    # prep a mask of the valid pixels, so we can blacken the invalids later
    mask = torch.max(im, dim=1, keepdim=True)[0]

    # turn the S dim into an explicit sequence
    coeffs = np.linspace(1.0, float(S), S).astype(np.float32) / float(S)

    # # increase the spacing from the center
    # coeffs[:int(S/2)] -= 2.0
    # coeffs[int(S/2)+1:] += 2.0

    coeffs = torch.from_numpy(coeffs).float().to(im.device)
    coeffs = coeffs.reshape(1, S, 1, 1).repeat(B, 1, H, W)
    # scale each channel by the right coeff
    im = im * coeffs
    # now im is in [1/S, 1], except for the invalid parts which are 0
    # keep the highest valid coeff at each pixel
    im = torch.max(im, dim=1, keepdim=True)[0]

    out = []
    for b in range(B):
        im_ = im[b]
        # move channels out to last dim_
        im_ = im_.detach().cpu().numpy()
        im_ = np.squeeze(im_)
        # im_ is H x W
        if colormap == 'coolwarm':
            im_ = cm.coolwarm(im_)[:, :, :3]
        elif colormap == 'PiYG':
            im_ = cm.PiYG(im_)[:, :, :3]
        elif colormap == 'winter':
            im_ = cm.winter(im_)[:, :, :3]
        elif colormap == 'spring':
            im_ = cm.spring(im_)[:, :, :3]
        elif colormap == 'onediff':
            im_ = np.reshape(im_, (-1))
            im0_ = cm.spring(im_)[:, :3]
            im1_ = cm.winter(im_)[:, :3]
            im1_[im_ == 1 / float(S)] = im0_[im_ == 1 / float(S)]
            im_ = np.reshape(im1_, (H, W, 3))
        else:
            assert (False)  # invalid colormap
        # move channels into dim 0
        im_ = np.transpose(im_, [2, 0, 1])
        im_ = torch.from_numpy(im_).float().to(im.device)
        out.append(im_)
    out = torch.stack(out, dim=0)

    # blacken the invalid pixels, instead of using the 0-color
    out = out * mask
    # out = out*255.0

    # put it in [-0.5, 0.5]
    out = out - 0.5

    return out


def colorize(d):
    # this is actually just grayscale right now

    if d.ndim == 2:
        d = d.unsqueeze(dim=0)
    else:
        assert (d.ndim == 3)
    # copy to the three chans
    d = d.repeat(3, 1, 1)
    return d


def oned2inferno(d, norm=True):
    # convert a 1chan input to a 3chan image output

    # if it's just B x H x W, add a C dim
    if d.ndim == 3:
        d = d.unsqueeze(dim=1)
    # d should be B x C x H x W, where C=1
    B, C, H, W = list(d.shape)
    assert (C == 1)

    if norm:
        d = sam_pt.point_tracker.utils.basic.normalize(d)

    rgb = torch.zeros(B, 3, H, W)
    for b in list(range(B)):
        rgb[b] = colorize(d[b])

    rgb = (255.0 * rgb).type(torch.ByteTensor)

    # rgb = tf.cast(255.0*rgb, tf.uint8)
    # rgb = tf.reshape(rgb, [-1, hyp.H, hyp.W, 3])
    # rgb = tf.expand_dims(rgb, axis=0)
    return rgb


def draw_frame_id_on_vis(vis, frame_id, font_scale=0.5, left=5, top=20, font_thickness=1):
    rgb = vis.detach().cpu().numpy()[0]
    rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    font_color = (255, 255, 255)
    # print('putting frame id', frame_id)

    frame_str = sam_pt.point_tracker.utils.basic.strnum(frame_id) if type(frame_id) != str else frame_id

    # Write each line of text in a new row
    (_, label_height), _ = cv2.getTextSize(frame_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    for i, line in enumerate(frame_str.split('\n')):
        top_i = top + i * label_height
        cv2.putText(
            rgb,
            line,
            (left, top_i),  # from left, from top
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_color,
            font_thickness,
        )

    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    vis = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return vis


COLORMAP_FILE = "./sam_pt/utils/bremm.png"


class ColorMap2d:
    def __init__(self, filename=None):
        self._colormap_file = filename or COLORMAP_FILE
        self._img = plt.imread(self._colormap_file)

        self._height = self._img.shape[0]
        self._width = self._img.shape[1]

    def __call__(self, X):
        assert len(X.shape) == 2
        output = np.zeros((X.shape[0], 3))
        for i in range(X.shape[0]):
            x, y = X[i, :]
            xp = int(self._width * x)
            yp = int(self._height * y)
            output[i, :] = self._img[yp, xp]
        return output


def get_n_colors(N, sequential=False):
    label_colors = []
    for ii in range(N):
        if sequential:
            rgb = cm.winter(ii / (N - 1))
            rgb = (np.array(rgb) * 255).astype(np.uint8)[:3]
        else:
            rgb = np.zeros(3)
            while np.sum(rgb) < 128:  # ensure min brightness
                rgb = np.random.randint(0, 256, 3)
        label_colors.append(rgb)
    return label_colors


class Summ_writer(object):
    def __init__(self, writer, global_step, log_freq=10, fps=8, scalar_freq=100, just_gif=False):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = log_freq
        self.fps = fps
        self.just_gif = just_gif
        self.maxwidth = 10000
        self.save_this = (self.global_step % self.log_freq == 0)
        self.scalar_freq = max(scalar_freq, 1)

    def summ_gif(self, name, tensor, blacken_zeros=False):
        # tensor should be in B x S x C x H x W

        assert tensor.dtype in {torch.uint8, torch.float32}
        shape = list(tensor.shape)

        if tensor.dtype == torch.float32:
            tensor = back2color(tensor, blacken_zeros=blacken_zeros)

        video_to_write = tensor[0:1]

        S = video_to_write.shape[1]
        if S == 1:
            # video_to_write is 1 x 1 x C x H x W
            self.writer.add_image(name, video_to_write[0, 0], global_step=self.global_step)
        else:
            self.writer.add_video(name, video_to_write, fps=self.fps, global_step=self.global_step)

        return video_to_write

    def summ_rgbs(self, name, ims, frame_ids=None, blacken_zeros=False, only_return=False):
        if self.save_this:

            ims = gif_and_tile(ims, just_gif=self.just_gif)
            vis = ims

            assert vis.dtype in {torch.uint8, torch.float32}

            if vis.dtype == torch.float32:
                vis = back2color(vis, blacken_zeros)

            B, S, C, H, W = list(vis.shape)

            if frame_ids is not None:
                assert (len(frame_ids) == S)
                for s in range(S):
                    vis[:, s] = draw_frame_id_on_vis(vis[:, s], frame_ids[s])

            if int(W) > self.maxwidth:
                vis = vis[:, :, :, :self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis, blacken_zeros)

    def summ_rgb(self, name, ims, blacken_zeros=False, frame_id=None, only_return=False, halfres=False):
        if self.save_this:
            assert ims.dtype in {torch.uint8, torch.float32}

            if ims.dtype == torch.float32:
                ims = back2color(ims, blacken_zeros)

            # ims is B x C x H x W
            vis = ims[0:1]  # just the first one
            B, C, H, W = list(vis.shape)

            if halfres:
                vis = F.interpolate(vis, scale_factor=0.5)

            if frame_id is not None:
                vis = draw_frame_id_on_vis(vis, frame_id)

            if int(W) > self.maxwidth:
                vis = vis[:, :, :, :self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis.unsqueeze(1), blacken_zeros)

    def summ_flow(self, name, im, clip=0.0, only_return=False, frame_id=None):
        # flow is B x C x D x W
        if self.save_this:
            return self.summ_rgb(name, self.flow2color(im, clip=clip), only_return=only_return, frame_id=frame_id)
        else:
            return None

    def summ_oneds(self, name, ims, frame_ids=None, bev=False, fro=False, logvis=False, reduce_max=False, max_val=0.0,
                   norm=True, only_return=False):
        if self.save_this:
            if bev:
                B, C, H, _, W = list(ims[0].shape)
                if reduce_max:
                    ims = [torch.max(im, dim=3)[0] for im in ims]
                else:
                    ims = [torch.mean(im, dim=3) for im in ims]
            elif fro:
                B, C, _, H, W = list(ims[0].shape)
                if reduce_max:
                    ims = [torch.max(im, dim=2)[0] for im in ims]
                else:
                    ims = [torch.mean(im, dim=2) for im in ims]

            if len(ims) != 1:  # sequence
                im = gif_and_tile(ims, just_gif=self.just_gif)
            else:
                im = torch.stack(ims, dim=1)  # single frame

            B, S, C, H, W = list(im.shape)

            if logvis and max_val:
                max_val = np.log(max_val)
                im = torch.log(torch.clamp(im, 0) + 1.0)
                im = torch.clamp(im, 0, max_val)
                im = im / max_val
                norm = False
            elif max_val:
                im = torch.clamp(im, 0, max_val)
                im = im / max_val
                norm = False

            if norm:
                # normalize before oned2inferno,
                # so that the ranges are similar within B across S
                im = sam_pt.point_tracker.utils.basic.normalize(im)

            im = im.view(B * S, C, H, W)
            vis = oned2inferno(im, norm=norm)
            vis = vis.view(B, S, 3, H, W)

            if frame_ids is not None:
                assert (len(frame_ids) == S)
                for s in range(S):
                    vis[:, s] = draw_frame_id_on_vis(vis[:, s], frame_ids[s])

            if W > self.maxwidth:
                vis = vis[..., :self.maxwidth]

            if only_return:
                return vis
            else:
                self.summ_gif(name, vis)

    def summ_oned(self, name, im, bev=False, fro=False, logvis=False, max_val=0, max_along_y=False, norm=True,
                  frame_id=None, only_return=False):
        if self.save_this:

            if bev:
                B, C, H, _, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=3)[0]
                else:
                    im = torch.mean(im, dim=3)
            elif fro:
                B, C, _, H, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=2)[0]
                else:
                    im = torch.mean(im, dim=2)
            else:
                B, C, H, W = list(im.shape)

            im = im[0:1]  # just the first one
            assert (C == 1)

            if logvis and max_val:
                max_val = np.log(max_val)
                im = torch.log(im)
                im = torch.clamp(im, 0, max_val)
                im = im / max_val
                norm = False
            elif max_val:
                im = torch.clamp(im, 0, max_val) / max_val
                norm = False

            vis = oned2inferno(im, norm=norm)
            if W > self.maxwidth:
                vis = vis[..., :self.maxwidth]
            return self.summ_rgb(name, vis, blacken_zeros=False, frame_id=frame_id, only_return=only_return)

    def summ_feats(self, name, feats, valids=None, pca=True, fro=False, only_return=False, frame_ids=None):
        if self.save_this:
            if valids is not None:
                valids = torch.stack(valids, dim=1)

            feats = torch.stack(feats, dim=1)
            # feats leads with B x S x C

            if feats.ndim == 6:

                # feats is B x S x C x D x H x W
                if fro:
                    reduce_dim = 3
                else:
                    reduce_dim = 4

                if valids is None:
                    feats = torch.mean(feats, dim=reduce_dim)
                else:
                    valids = valids.repeat(1, 1, feats.size()[2], 1, 1, 1)
                    feats = sam_pt.point_tracker.utils.basic.reduce_masked_mean(feats, valids, dim=reduce_dim)

            B, S, C, D, W = list(feats.size())

            if not pca:
                # feats leads with B x S x C
                feats = torch.mean(torch.abs(feats), dim=2, keepdims=True)
                # feats leads with B x S x 1
                feats = torch.unbind(feats, dim=1)
                return self.summ_oneds(name=name, ims=feats, norm=True, only_return=only_return, frame_ids=frame_ids)

            else:
                __p = lambda x: sam_pt.point_tracker.utils.basic.pack_seqdim(x, B)
                __u = lambda x: sam_pt.point_tracker.utils.basic.unpack_seqdim(x, B)

                feats_ = __p(feats)

                if valids is None:
                    feats_pca_ = get_feat_pca(feats_)
                else:
                    valids_ = __p(valids)
                    feats_pca_ = get_feat_pca(feats_, valids)

                feats_pca = __u(feats_pca_)

                return self.summ_rgbs(name=name, ims=torch.unbind(feats_pca, dim=1), only_return=only_return,
                                      frame_ids=frame_ids)

    def summ_feat(self, name, feat, valid=None, pca=True, only_return=False, bev=False, fro=False, frame_id=None):
        if self.save_this:
            if feat.ndim == 5:  # B x C x D x H x W

                if bev:
                    reduce_axis = 3
                elif fro:
                    reduce_axis = 2
                else:
                    # default to bev
                    reduce_axis = 3

                if valid is None:
                    feat = torch.mean(feat, dim=reduce_axis)
                else:
                    valid = valid.repeat(1, feat.size()[1], 1, 1, 1)
                    feat = sam_pt.point_tracker.utils.basic.reduce_masked_mean(feat, valid, dim=reduce_axis)

            B, C, D, W = list(feat.shape)

            if not pca:
                feat = torch.mean(torch.abs(feat), dim=1, keepdims=True)
                # feat is B x 1 x D x W
                return self.summ_oned(name=name, im=feat, norm=True, only_return=only_return, frame_id=frame_id)
            else:
                feat_pca = get_feat_pca(feat, valid)
                return self.summ_rgb(name, feat_pca, only_return=only_return, frame_id=frame_id)

    def summ_scalar(self, name, value):
        if (not (isinstance(value, int) or isinstance(value, float) or isinstance(value, np.float32))) and (
                'torch' in value.type()):
            value = value.detach().cpu().numpy()
        if not np.isnan(value):
            if (self.log_freq == 1):
                self.writer.add_scalar(name, value, global_step=self.global_step)
            elif self.save_this or np.mod(self.global_step, self.scalar_freq) == 0:
                self.writer.add_scalar(name, value, global_step=self.global_step)

    def summ_traj2ds_on_rgbs(self, name, trajs, rgbs, valids=None, frame_ids=None, only_return=False, show_dots=True,
                             cmap='coolwarm', linewidth=1):
        # trajs is B, S, N, 2
        # rgbs is B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        B, S2, N, D = trajs.shape
        assert (S == S2)

        rgbs = rgbs[0]  # S, C, H, W
        trajs = trajs[0]  # S, N, 2
        if valids is None:
            valids = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            valids = valids[0]
        # print('trajs', trajs.shape)
        # print('valids', valids.shape)

        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(rgb).detach().cpu().numpy()
            rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
            rgbs_color.append(rgb)  # each element 3 x H x W

        for i in range(N):
            if cmap == 'onediff' and i == 0:
                cmap_ = 'spring'
            elif cmap == 'onediff':
                cmap_ = 'winter'
            else:
                cmap_ = cmap
            traj = trajs[:, i].long().detach().cpu().numpy()  # S, 2
            valid = valids[:, i].long().detach().cpu().numpy()  # S

            # print('traj', traj.shape)
            # print('valid', valid.shape)

            for t in range(S):
                if valid[t]:
                    rgbs_color[t] = self.draw_traj_on_image_py(rgbs_color[t], traj[:t + 1], S=S, show_dots=show_dots,
                                                               cmap=cmap_, linewidth=linewidth)

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(name, rgbs, only_return=only_return, frame_ids=frame_ids)

    def summ_traj2ds_on_rgbs2(self, name, trajs, visibles, rgbs, valids=None, frame_ids=None, only_return=False,
                              show_dots=True, cmap='coolwarm', linewidth=1):
        # trajs is B, S, N, 2
        # rgbs is B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        B, S2, N, D = trajs.shape
        assert (S == S2)

        rgbs = rgbs[0]  # S, C, H, W
        trajs = trajs[0]  # S, N, 2
        visibles = visibles[0]  # S, N
        if valids is None:
            valids = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            valids = valids[0]
        # print('trajs', trajs.shape)
        # print('valids', valids.shape)

        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(rgb).detach().cpu().numpy()
            rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
            rgbs_color.append(rgb)  # each element 3 x H x W

        trajs = trajs.long().detach().cpu().numpy()  # S, N, 2
        visibles = visibles.float().detach().cpu().numpy()  # S, N
        valids = valids.long().detach().cpu().numpy()  # S, N

        for i in range(N):
            if cmap == 'onediff' and i == 0:
                cmap_ = 'spring'
            elif cmap == 'onediff':
                cmap_ = 'winter'
            else:
                cmap_ = cmap
            traj = trajs[:, i]  # S,2
            vis = visibles[:, i]  # S
            valid = valids[:, i]  # S
            rgbs_color = self.draw_traj_on_images_py(rgbs_color, traj, S=S, show_dots=show_dots, cmap=cmap_,
                                                     linewidth=linewidth)

        for i in range(N):
            if cmap == 'onediff' and i == 0:
                cmap_ = 'spring'
            elif cmap == 'onediff':
                cmap_ = 'winter'
            else:
                cmap_ = cmap
            traj = trajs[:, i]  # S,2
            vis = visibles[:, i]  # S
            valid = valids[:, i]  # S
            rgbs_color = self.draw_circ_on_images_py(rgbs_color, traj, vis, S=S, show_dots=show_dots, cmap=cmap_,
                                                     linewidth=linewidth)

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(name, rgbs, only_return=only_return, frame_ids=frame_ids)

    def summ_traj2ds_on_rgb(self, name, trajs, rgb, valids=None, show_dots=True, frame_id=None, only_return=False,
                            cmap='coolwarm', linewidth=1):
        # trajs is B, S, N, 2
        # rgb is B, C, H, W
        B, C, H, W = rgb.shape
        B, S, N, D = trajs.shape

        rgb = rgb[0]  # S, C, H, W
        trajs = trajs[0]  # S, N, 2

        if valids is None:
            valids = torch.ones_like(trajs[:, :, 0])
        else:
            valids = valids[0]

        rgb_color = back2color(rgb).detach().cpu().numpy()
        rgb_color = np.transpose(rgb_color, [1, 2, 0])  # put channels last

        # using maxdist will dampen the colors for short motions
        norms = torch.sqrt(1e-4 + torch.sum((trajs[-1] - trajs[0]) ** 2, dim=1))  # N
        maxdist = torch.quantile(norms, 0.95).detach().cpu().numpy()
        maxdist = None
        trajs = trajs.long().detach().cpu().numpy()  # S, N, 2
        valids = valids.long().detach().cpu().numpy()  # S, N

        for i in range(N):
            if cmap == 'onediff' and i == 0:
                cmap_ = 'spring'
            elif cmap == 'onediff':
                cmap_ = 'winter'
            else:
                cmap_ = cmap
            traj = trajs[:, i]  # S, 2
            valid = valids[:, i]  # S
            if valid[0] == 1:
                traj = traj[valid > 0]
                rgb_color = self.draw_traj_on_image_py(rgb_color, traj, S=S, show_dots=show_dots, cmap=cmap_,
                                                       maxdist=maxdist, linewidth=linewidth)

        rgb_color = torch.from_numpy(rgb_color).permute(2, 0, 1).unsqueeze(0)
        rgb = preprocess_color(rgb_color)
        return self.summ_rgb(name, rgb, only_return=only_return, frame_id=frame_id)

    def draw_traj_on_image_py(self, rgb, traj, S=50, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
        # all inputs are numpy tensors
        # rgb is 3 x H x W
        # traj is S x 2

        H, W, C = rgb.shape
        assert (C == 3)

        rgb = rgb.astype(np.uint8).copy()

        S1, D = traj.shape
        assert (D == 2)

        color_map = cm.get_cmap(cmap)
        S1, D = traj.shape

        for s in range(S1 - 1):
            if maxdist is not None:
                val = (np.sqrt(np.sum((traj[s] - traj[0]) ** 2)) / maxdist).clip(0, 1)
                color = np.array(color_map(val)[:3]) * 255  # rgb
            else:
                color = np.array(color_map((s) / max(1, float(S - 2)))[:3]) * 255  # rgb

            cv2.line(rgb,
                     (int(traj[s, 0]), int(traj[s, 1])),
                     (int(traj[s + 1, 0]), int(traj[s + 1, 1])),
                     color,
                     linewidth,
                     cv2.LINE_AA)
            if show_dots:
                cv2.circle(rgb, (traj[s, 0], traj[s, 1]), linewidth, color, -1)

        if maxdist is not None:
            val = (np.sqrt(np.sum((traj[-1] - traj[0]) ** 2)) / maxdist).clip(0, 1)
            color = np.array(color_map(val)[:3]) * 255  # rgb
        else:
            # draw the endpoint of traj, using the next color (which may be the last color)
            color = np.array(color_map((S1 - 1) / max(1, float(S - 2)))[:3]) * 255  # rgb

        # color = np.array(color_map(1.0)[:3]) * 255
        cv2.circle(rgb, (traj[-1, 0], traj[-1, 1]), linewidth * 2, color, -1)

        return rgb

    def draw_traj_on_images_py(self, rgbs, traj, S=50, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
        # all inputs are numpy tensors
        # rgbs is a list of H,W,3
        # traj is S,2
        H, W, C = rgbs[0].shape
        assert (C == 3)

        rgbs = [rgb.astype(np.uint8).copy() for rgb in rgbs]

        S1, D = traj.shape
        assert (D == 2)

        x = int(np.clip(traj[0, 0], 0, W - 1))
        y = int(np.clip(traj[0, 1], 0, H - 1))
        color = rgbs[0][y, x]
        color = (int(color[0]), int(color[1]), int(color[2]))
        for s in range(S):
            # bak_color = np.array(color_map(1.0)[:3]) * 255 # rgb
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*4, bak_color, -1)
            cv2.polylines(rgbs[s],
                          [traj[:s + 1]],
                          False,
                          color,
                          linewidth,
                          cv2.LINE_AA)
        return rgbs

    def draw_circ_on_images_py(self, rgbs, traj, vis, S=50, linewidth=1, show_dots=False, cmap='coolwarm',
                               maxdist=None):
        # all inputs are numpy tensors
        # rgbs is a list of 3,H,W
        # traj is S,2
        H, W, C = rgbs[0].shape
        assert (C == 3)

        rgbs = [rgb.astype(np.uint8).copy() for rgb in rgbs]

        S1, D = traj.shape
        assert (D == 2)

        bremm = ColorMap2d()
        traj_ = traj[0:1].astype(np.float32)
        traj_[:, 0] /= float(W)
        traj_[:, 1] /= float(H)
        color = bremm(traj_)
        # print('color', color)
        color = (color[0] * 255).astype(np.uint8)
        # print('color', color)
        color = (int(color[0]), int(color[1]), int(color[2]))

        x = int(np.clip(traj[0, 0], 0, W - 1))
        y = int(np.clip(traj[0, 1], 0, H - 1))
        color_ = rgbs[0][y, x]
        color_ = (int(color_[0]), int(color_[1]), int(color_[2]))
        color_ = (int(color_[0]), int(color_[1]), int(color_[2]))
        # print('color_', color_)
        # print('vis[0]', vis[0])
        for s in range(S):
            cv2.circle(rgbs[s], (traj[s, 0], traj[s, 1]), linewidth * 4, color, -1)
            vis_color = int(np.squeeze(vis[s]) * 255)
            vis_color = (vis_color, vis_color, vis_color)
            cv2.circle(rgbs[s], (traj[s, 0], traj[s, 1]), linewidth * 2, vis_color, -1)

        return rgbs


def erode2d(im, times=1, device='cuda'):
    weights2d = torch.ones(1, 1, 3, 3, device=device)
    for time in range(times):
        im = 1.0 - F.conv2d(1.0 - im, weights2d, padding=1).clamp(0, 1)
    return im
