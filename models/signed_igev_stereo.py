"""IGEV-Stereo variant with a signed GWC cost volume over [d_min, d_max].

The upstream IGEV GWC volume is built only over non-negative disparities
[0, max_disp/4], so out-of-bounds lookups in the negative regime return
zeros via grid_sample. This subclass extends the volume's disparity axis
to a signed range; the init disparity comes from a soft-argmax over the
signed bins, and the recurrent geometry-feature lookup is offset by
d_min so bin 0 of the volume corresponds to disparity d_min.

All disparity values throughout the GRU loop are in 1/4-resolution units
(matching upstream IGEV's convention).
"""
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IGEV_ROOT = Path(os.environ.get("IGEV_ROOT", os.path.expanduser("~/git/IGEV-clone/IGEV-Stereo")))
if str(IGEV_ROOT) not in sys.path:
    sys.path.insert(0, str(IGEV_ROOT))
if str(IGEV_ROOT / "core") not in sys.path:
    sys.path.insert(0, str(IGEV_ROOT / "core"))

from core.igev_stereo import IGEVStereo, autocast
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import groupwise_correlation, context_upsample
from core.utils.utils import bilinear_sampler


def build_gwc_volume_signed(refimg_fea, targetimg_fea, d_min: int, d_max: int, num_groups: int):
    """GWC volume over signed disparity bins [d_min, d_max] (inclusive).

    Bin index `idx` corresponds to disparity `d_min + idx`.
    For d > 0: left[x] matches right[x-d] (uncrossed / positive disparity).
    For d < 0: left[x] matches right[x+|d|] = right[x-d] (crossed / negative).
    """
    B, C, H, W = refimg_fea.shape
    n_disp = d_max - d_min + 1
    volume = refimg_fea.new_zeros([B, num_groups, n_disp, H, W])
    for idx, d in enumerate(range(d_min, d_max + 1)):
        if d > 0:
            volume[:, :, idx, :, d:] = groupwise_correlation(
                refimg_fea[:, :, :, d:], targetimg_fea[:, :, :, :-d], num_groups)
        elif d < 0:
            nd = -d
            volume[:, :, idx, :, :-nd] = groupwise_correlation(
                refimg_fea[:, :, :, :-nd], targetimg_fea[:, :, :, nd:], num_groups)
        else:
            volume[:, :, idx, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    return volume.contiguous()


def disparity_regression_signed(prob, d_min: int, d_max: int):
    n_disp = d_max - d_min + 1
    disp_values = torch.arange(d_min, d_max + 1, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.view(1, n_disp, 1, 1)
    return torch.sum(prob * disp_values, 1, keepdim=True)


class SignedCombinedGeoVolume(Combined_Geo_Encoding_Volume):
    """Same as upstream IGEV `Combined_Geo_Encoding_Volume` but the GWC stream
    is indexed with a `d_min` offset: bin 0 of `geo_volume_pyramid[i]`
    corresponds to disparity `d_min` (quarter-res units), not 0.

    The init_corr (all-pairs) stream is unchanged: it queries the right image
    at `coords - disp + dx`, which is sign-agnostic.
    """

    def __init__(self, init_fmap1, init_fmap2, geo_volume, d_min_quarter: int,
                 num_levels=2, radius=4):
        super().__init__(init_fmap1, init_fmap2, geo_volume,
                         num_levels=num_levels, radius=radius)
        self.d_min_quarter = d_min_quarter

    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1).view(1, 1, 2 * r + 1, 1).to(disp.device)
            # bin index = (disp - d_min_quarter) / 2**i ; add radius offsets dx
            x0 = dx + (disp.reshape(b * h * w, 1, 1, 1) - self.d_min_quarter) / 2 ** i
            y0 = torch.zeros_like(x0)
            disp_lvl = torch.cat([x0, y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.reshape(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = (coords.reshape(b * h * w, 1, 1, 1) / 2 ** i
                       - disp.reshape(b * h * w, 1, 1, 1) / 2 ** i
                       + dx)
            init_coords_lvl = torch.cat([init_x0, y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.reshape(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out_pyramid = torch.cat(out_pyramid, dim=-1)
        return out_pyramid.permute(0, 3, 1, 2).contiguous()


class SignedIGEVStereo(IGEVStereo):
    """IGEVStereo with signed GWC volume.

    `d_neg`, `d_pos` are FULL-resolution disparity bounds (e.g., 64 and 192).
    Internally the volume operates at 1/4 resolution, so bin range is
    [d_min_quarter, d_max_quarter] = [-d_neg/4, d_pos/4 - 1]. d_neg and d_pos
    must both be divisible by 4.
    """

    def __init__(self, args, d_neg: int = 64, d_pos: int = 192):
        super().__init__(args)
        assert d_neg % 4 == 0 and d_pos % 4 == 0
        self.d_neg = d_neg
        self.d_pos = d_pos
        self.d_min_quarter = -(d_neg // 4)
        self.d_max_quarter = (d_pos // 4) - 1

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision,
                      dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            # SIGNED GWC build
            gwc_volume = build_gwc_volume_signed(match_left, match_right,
                                                 self.d_min_quarter, self.d_max_quarter, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

            # Init disp from geometry encoding volume (signed soft-argmax)
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression_signed(prob, self.d_min_quarter, self.d_max_quarter)
            del prob, gwc_volume

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                        for i, conv in zip(inp_list, self.context_zqr_convs)]

        # SIGNED geometry lookup
        geo_fn = SignedCombinedGeoVolume(
            match_left.float(), match_right.float(), geo_encoding_volume.float(),
            d_min_quarter=self.d_min_quarter,
            num_levels=self.args.corr_levels,
            radius=self.args.corr_radius,
        )
        b, c, h, w = match_left.shape
        coords = (torch.arange(w).float().to(match_left.device)
                  .reshape(1, 1, w, 1).repeat(b, h, 1, 1))
        disp = init_disp
        disp_preds = []

        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision,
                          dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list, inp_list, geo_feat, disp,
                    iter16=self.args.n_gru_layers == 3,
                    iter08=self.args.n_gru_layers >= 2,
                )
            disp = disp + delta_disp
            if test_mode and itr < iters - 1:
                continue
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up
        # Upsample init_disp from 1/4-res signed quarter units to full-res signed
        # disparity (matches upstream IGEV's training-mode return convention).
        init_disp = context_upsample(init_disp * 4.0, spx_pred.float()).unsqueeze(1)
        return init_disp, disp_preds
