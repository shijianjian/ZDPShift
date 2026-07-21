"""
Crawl download.blender.org demo directories, list every .blend, and
download the viable ones into scenes/.

Filters:
  - skip < 50 KB (probably empty placeholder)
  - skip > 200 MB (too slow to render at scale)
  - skip names hinting at gpencil / animation / physics-particle
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin


BASE = "https://download.blender.org/demo/"

DIRS_TO_CRAWL = [
    "splash/", "cycles/", "eevee/", "old_demos/", "test/",
    "physics/", "sculpt_mode/", "geometry-nodes/", "movies/",
    "asset-bundles/cube-diorama/", "asset-bundles/ellie-pose-library/",
]

SKIP_NAME_HINTS = ("gpencil", "_anim", "particle")


def curl_get(url, timeout=15):
    r = subprocess.run(["curl", "-sL", "--max-time", str(timeout), url],
                       capture_output=True)
    return r.stdout.decode("utf8", errors="ignore")


def curl_head_size(url, timeout=8):
    r = subprocess.run(
        ["curl", "-sIL", "--max-time", str(timeout), url],
        capture_output=True,
    )
    out = r.stdout.decode("utf8", errors="ignore")
    sizes = re.findall(r"(?im)^Content-Length:\s*(\d+)", out)
    return int(sizes[-1]) if sizes else -1


def list_candidates(min_size=50_000, max_size=200_000_000):
    found = []
    for d in DIRS_TO_CRAWL:
        url = urljoin(BASE, d)
        html = curl_get(url)
        for m in re.finditer(r'href="([^"]+\.blend)"', html):
            fname = m.group(1)
            if any(h in fname.lower() for h in SKIP_NAME_HINTS):
                continue
            full = urljoin(url, fname)
            size = curl_head_size(full)
            if size < 0:
                continue
            ok = min_size <= size <= max_size
            found.append((d, fname, size, full, ok))
    return found


def download(url: str, dest: Path, timeout=300):
    if dest.exists():
        return "skip-exists"
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = subprocess.run(
        ["curl", "-sL", "--max-time", str(timeout), "-o", str(tmp), url],
        capture_output=True,
    )
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
        if tmp.exists():
            tmp.unlink()
        return f"fail:{r.returncode}"
    tmp.rename(dest)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scenes/")
    ap.add_argument("--min-size", type=int, default=50_000)
    ap.add_argument("--max-size", type=int, default=200_000_000)
    ap.add_argument("--list-only", action="store_true",
                    help="Only list candidates without downloading")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Crawling {len(DIRS_TO_CRAWL)} dirs...")
    cands = list_candidates(args.min_size, args.max_size)
    cands.sort(key=lambda x: x[2])

    print(f"\nFound {len(cands)} .blend listings ({sum(1 for c in cands if c[4])} match size filter):")
    print(f"{'dir':<35} {'name':<55} {'size MB':>10}  {'kept?':>6}")
    print("-" * 115)
    for d, n, sz, _, ok in cands:
        print(f"{d:<35} {n:<55} {sz/1e6:>10.2f}  {'YES' if ok else '  '}")

    keep = [c for c in cands if c[4]]
    print(f"\n=> {len(keep)} candidate scenes, total {sum(c[2] for c in keep)/1e6:.1f} MB")

    if args.list_only:
        return

    print(f"\nDownloading to {out_dir} ...")
    t0 = time.time()
    for d, name, sz, url, ok in keep:
        # Use a deterministic file name
        local = out_dir / name
        status = download(url, local)
        size_disp = f"{sz/1e6:>6.1f} MB"
        print(f"  [{status:>12}]  {size_disp}  {name}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
