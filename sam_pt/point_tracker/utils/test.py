# Taken from: https://github.com/aharley/pips/blob/486124b4236bb228a20750b496f0fa8aa6343157/utils/test.py

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def prep_frame_for_dino(img, scale_size=[192]):
    """
    read a single frame & preprocess
    """
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if (ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()

    def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
        return x

    img = color_normalize(img)
    return img, ori_h, ori_w


def get_feats_from_dino(model, frame):
    # batch version of the other func
    B = frame.shape[0]
    patch_size = model.patch_embed.patch_size
    h, w = int(frame.shape[2] / patch_size), int(frame.shape[3] / patch_size)
    out = model.get_intermediate_layers(frame.cuda(), n=1)[0]  # B, 1+h*w, dim
    dim = out.shape[-1]
    out = out[:, 1:, :]  # discard the [CLS] token
    outmap = out.permute(0, 2, 1).reshape(B, dim, h, w)
    return out, outmap, h, w


def restrict_neighborhood(h, w):
    size_mask_neighborhood = 12
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def label_propagation(h, w, feat_tar, list_frame_feats, list_segs, mask_neighborhood=None):
    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats)  # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    # print('feat_tar', feat_tar.shape)
    # print('feat_sources', feat_sources.shape)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1)

    size_mask_neighborhood = 12
    if size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h * w)  # nmb_context*h*w (source: keys) x h*w (tar: queries)
    topk = 5
    tk_val, _ = torch.topk(aff, dim=0, k=topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T  # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)

    return seg_tar, mask_neighborhood


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if (mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


def get_dino_output(dino, rgbs, trajs_g, vis_g):
    B, S, C, H, W = rgbs.shape

    B1, S1, N, D = trajs_g.shape
    assert (B1 == B)
    assert (S1 == S)
    assert (D == 2)

    assert (B == 1)
    xy0 = trajs_g[:, 0]  # B, N, 2

    # The queue stores the n preceeding frames
    import queue
    import copy
    n_last_frames = 7
    que = queue.Queue(n_last_frames)

    # run dino
    prep_rgbs = []
    for s in range(S):
        prep_rgb, ori_h, ori_w = prep_frame_for_dino(rgbs[0, s].permute(1, 2, 0).detach().cpu().numpy(), scale_size=[H])
        prep_rgbs.append(prep_rgb)
    prep_rgbs = torch.stack(prep_rgbs, dim=0)  # S, 3, H, W
    with torch.no_grad():
        bs = 8
        idx = 0
        featmaps = []
        while idx < S:
            end_id = min(S, idx + bs)
            _, featmaps_cur, h, w = get_feats_from_dino(dino, prep_rgbs[idx:end_id])  # S, C, h, w
            idx = end_id
            featmaps.append(featmaps_cur)
        featmaps = torch.cat(featmaps, dim=0)
    C = featmaps.shape[1]
    featmaps = featmaps.unsqueeze(0)  # 1, S, C, h, w
    # featmaps = F.normalize(featmaps, dim=2, p=2)

    xy0 = trajs_g[:, 0, :]  # B, N, 2
    patch_size = dino.patch_embed.patch_size
    first_seg = torch.zeros((1, N, H // patch_size, W // patch_size))
    for n in range(N):
        first_seg[0, n, (xy0[0, n, 1] / patch_size).long(), (xy0[0, n, 0] / patch_size).long()] = 1

    frame1_feat = featmaps[0, 0].reshape(C, h * w)  # dim x h*w
    mask_neighborhood = None
    accs = []
    trajs_e = torch.zeros_like(trajs_g)
    trajs_e[0, 0] = trajs_g[0, 0]
    for cnt in range(1, S):
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        feat_tar = featmaps[0, cnt].reshape(C, h * w)

        frame_tar_avg, mask_neighborhood = label_propagation(h, w, feat_tar.T, used_frame_feats, used_segs,
                                                             mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=patch_size, mode='bilinear', align_corners=False,
                                      recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        for n in range(N):
            vis = vis_g[0, cnt, n]
            if len(torch.nonzero(frame_tar_avg[n])) > 0:
                # weighted average
                nz = torch.nonzero(frame_tar_avg[n])
                coord_e = torch.sum(frame_tar_avg[n][nz[:, 0], nz[:, 1]].reshape(-1, 1) * nz.float(), 0) / \
                          frame_tar_avg[n][nz[:, 0], nz[:, 1]].sum()  # 2
                coord_e = coord_e[[1, 0]]
            else:
                # stay where it was
                coord_e = trajs_e[0, cnt - 1, n]

            trajs_e[0, cnt, n] = coord_e
    return trajs_e
