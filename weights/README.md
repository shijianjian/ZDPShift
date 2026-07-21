# Pretrained weights

Fine-tuned ZDPShift checkpoints (to be uploaded on release):

| File | Backbone | Route | Notes |
|------|----------|-------|-------|
| `raft_zdpshift.pth`        | RAFT-Stereo       | data recipe          | mean EPE 0.97 px |
| `igev_zdpshift_signed.pth` | IGEV-Stereo       | + signed cost volume | mean EPE 0.92 px |
| `fs_zdpshift_signed.pth`   | FoundationStereo  | + signed cost volume | mean EPE 0.76 px (best) |

Download (link added on release):

```bash
# placeholder — replace with the real release URL
# wget -P weights/ https://<host>/zdpshift/raft_zdpshift.pth
```

The `d_neg=64`, `d_pos=192` signed-volume settings used for the checkpoints are
the defaults in `eval_foundation_stereo.py` / `eval_igev_signed.py`.
