"""FoundationStereo variant with a signed GWC + concat cost volume over
[d_min, d_max], matching the SignedIGEVStereo design.

Upstream FoundationStereo's cost volumes are built only over non-negative
disparities [0, max_disp/4]; geometry-lookup grid_sample then returns zeros
for any disparity outside that range. This subclass extends the volume's
disparity axis to a signed range, soft-argmax's init disparity over signed
bins, and offsets the geometry-feature lookup by d_min so bin 0 corresponds
to disparity d_min.

The DINOv2 ViT-L backbone is left frozen (upstream `Feature.__init__`
already calls `freeze_model(self.dino)`; we don't touch it).

All quarter-resolution disparity bookkeeping mirrors the original.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

FS_DIR = Path(__file__).resolve().parent.parent / "third_party/FoundationStereo"
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

from core.foundation_stereo import FoundationStereo, autocast, normalize_image
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import groupwise_correlation, context_upsample
from core.utils.utils import bilinear_sampler


def build_gwc_volume_signed(refimg_fea, targetimg_fea, d_min: int, d_max: int, num_groups: int):
    """GWC volume indexed over signed disparity bins [d_min, d_max] (inclusive).

    Bin idx -> disparity = d_min + idx.
      d > 0: left[x] matches right[x - d]
      d < 0: left[x] matches right[x - d] = right[x + |d|]
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


def build_concat_volume_signed(refimg_fea, targetimg_fea, d_min: int, d_max: int):
    """Concat (left || right-shifted-by-d) volume over signed disparity bins."""
    B, C, H, W = refimg_fea.shape
    n_disp = d_max - d_min + 1
    volume = refimg_fea.new_zeros([B, 2 * C, n_disp, H, W])
    for idx, d in enumerate(range(d_min, d_max + 1)):
        if d > 0:
            volume[:, :C, idx, :, :] = refimg_fea
            volume[:, C:, idx, :, d:] = targetimg_fea[:, :, :, :-d]
        elif d < 0:
            nd = -d
            volume[:, :C, idx, :, :] = refimg_fea
            volume[:, C:, idx, :, :-nd] = targetimg_fea[:, :, :, nd:]
        else:
            volume[:, :C, idx, :, :] = refimg_fea
            volume[:, C:, idx, :, :] = targetimg_fea
    return volume.contiguous()


def disparity_regression_signed(prob, d_min: int, d_max: int):
    n_disp = d_max - d_min + 1
    disp_values = torch.arange(d_min, d_max + 1, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.reshape(1, n_disp, 1, 1)
    return torch.sum(prob * disp_values, 1, keepdim=True)


class SignedCombinedGeoVolume(Combined_Geo_Encoding_Volume):
    """Same as upstream FS `Combined_Geo_Encoding_Volume` but the GWC stream
    is indexed with a `d_min` offset: bin 0 of `geo_volume_pyramid[i]`
    corresponds to disparity `d_min` (quarter-res units), not 0.

    The init_corr (all-pairs) stream is unchanged: it queries the right image
    at `coords - disp + dx`, which is sign-agnostic.
    """

    def __init__(self, init_fmap1, init_fmap2, geo_volume, d_min_quarter: int,
                 num_levels=2, dx=None):
        super().__init__(init_fmap1, init_fmap2, geo_volume,
                         num_levels=num_levels, dx=dx)
        self.d_min_quarter = d_min_quarter

    def __call__(self, disp, coords, low_memory=False):
        b, _, h, w = disp.shape
        self.dx = self.dx.to(disp.device)
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            # GWC bin index = (disp - d_min_quarter) / 2**i ; add radius offsets dx
            x0 = self.dx + (disp.reshape(b * h * w, 1, 1, 1) - self.d_min_quarter) / 2 ** i
            y0 = torch.zeros_like(x0)
            disp_lvl = torch.cat([x0, y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl, low_memory=low_memory)
            geo_volume = geo_volume.reshape(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = (coords.reshape(b * h * w, 1, 1, 1) / 2 ** i
                       - disp.reshape(b * h * w, 1, 1, 1) / 2 ** i
                       + self.dx)
            init_coords_lvl = torch.cat([init_x0, y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl, low_memory=low_memory)
            init_corr = init_corr.reshape(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out_pyramid = torch.cat(out_pyramid, dim=-1)
        return out_pyramid.permute(0, 3, 1, 2).contiguous()


class SignedFoundationStereo(FoundationStereo):
    """FoundationStereo with signed GWC + concat cost volumes.

    `d_neg`, `d_pos` are FULL-resolution disparity bounds (e.g., 64 and 192).
    Internally the volume operates at 1/4 resolution. Both must be divisible by 4.
    """

    def __init__(self, args, d_neg: int = 64, d_pos: int = 192):
        super().__init__(args)
        assert d_neg % 4 == 0 and d_pos % 4 == 0
        self.d_neg = d_neg
        self.d_pos = d_pos
        self.d_min_quarter = -(d_neg // 4)
        self.d_max_quarter = (d_pos // 4) - 1

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False,
                low_memory=False, init_disp=None):
        B = len(image1)
        low_memory = low_memory or (self.args.get("low_memory", False))
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)
        with autocast(enabled=self.args.mixed_precision):
            out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
            vit_feat = vit_feat[:B]
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            stem_2x = self.stem_2(image1)

            # SIGNED volumes
            gwc_volume = build_gwc_volume_signed(
                features_left[0], features_right[0],
                self.d_min_quarter, self.d_max_quarter, self.cv_group)
            left_tmp = self.proj_cmb(features_left[0])
            right_tmp = self.proj_cmb(features_right[0])
            concat_volume = build_concat_volume_signed(
                left_tmp, right_tmp, self.d_min_quarter, self.d_max_quarter)
            del left_tmp, right_tmp
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            comb_volume = self.corr_stem(comb_volume)
            comb_volume = self.corr_feature_att(comb_volume, features_left[0])
            comb_volume = self.cost_agg(comb_volume, features_left)

            # SIGNED soft-argmax for init disparity
            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)
            if init_disp is None:
                init_disp = disparity_regression_signed(
                    prob, self.d_min_quarter, self.d_max_quarter)

            cnet_list = self.cnet(image1, vit_feat=vit_feat, num_layers=self.args.n_gru_layers)
            cnet_list = list(cnet_list)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [self.cam(x) * x for x in inp_list]
            att = [self.sam(x) for x in inp_list]

        # SIGNED geometry lookup
        geo_fn = SignedCombinedGeoVolume(
            features_left[0].float(), features_right[0].float(),
            comb_volume.float(),
            d_min_quarter=self.d_min_quarter,
            num_levels=self.args.corr_levels,
            dx=self.dx,
        )
        b, c, h, w = features_left[0].shape
        coords = (torch.arange(w, dtype=torch.float, device=init_disp.device)
                  .reshape(1, 1, w, 1).repeat(b, h, 1, 1))
        disp = init_disp.float()
        disp_preds = []

        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, low_memory=low_memory)
            with autocast(enabled=self.args.mixed_precision):
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list, inp_list, geo_feat, disp, att)
            disp = disp + delta_disp.float()
            if test_mode and itr < iters - 1:
                continue
            disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up
        return init_disp, disp_preds
