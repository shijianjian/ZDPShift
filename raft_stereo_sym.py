"""Symmetric-cost-volume RAFT-Stereo wrapper.

Per the CorrBlock1D audit (docs/superpowers/specs/corr_audit_notes.md), the
upstream implementation already samples symmetrically around the current flow
estimate — `dx = linspace(-r, r, 2*r+1)`, no sign clamps on flow or coords1,
bilinear_sampler is sign-neutral. The vanilla model's failure on negative
disparity is therefore a training-distribution issue (the GRU has only ever
been gradient-rewarded for non-negative flow), not an architectural one.

SymRAFTStereo is intentionally identical to RAFTStereo. It exists as a separate
class so checkpoints from this experiment can be tagged distinctly from the
public Sceneflow weights, and so the experiment lineage is explicit.
"""
import sys
from pathlib import Path

THIRD = Path(__file__).resolve().parent / "third_party/RAFT-Stereo"
sys.path.insert(0, str(THIRD))
sys.path.insert(0, str(THIRD / "core"))

from core.raft_stereo import RAFTStereo


class SymRAFTStereo(RAFTStereo):
    """Alias of RAFTStereo. Identical behavior; renamed for experiment provenance."""
    pass
