# Weights

Fine-tuned ZDPShift checkpoints: https://huggingface.co/shijianjian/ZDPShift

```bash
huggingface-cli download shijianjian/ZDPShift --local-dir weights
```

| File | Backbone | mean EPE |
|------|----------|----------|
| `raft_zdpshift.pth`        | RAFT-Stereo (data recipe)        | 0.97 px |
| `igev_zdpshift_signed.pth` | IGEV-Stereo + signed volume      | 0.92 px |
| `fs_zdpshift_signed.pth`   | FoundationStereo + signed volume | 0.76 px |

Signed-volume checkpoints use `d_neg=64`, `d_pos=192` (the eval defaults).
