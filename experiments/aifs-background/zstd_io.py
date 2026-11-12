"""zstd-7 lossless containers for the AIFS pipeline data.

Formats (all bit-exact, verified lossless; ~1.8x on 0.25deg shards):
  <stem>.npy.zst : zstd-7 frame wrapping a complete standard .npy file
  <stem>.npz.zst : zstd-7 frame wrapping a complete UNCOMPRESSED .npz file
Readers fall back transparently to legacy .npy / .npz.

Requires numcodecs (present in both repo .venv and v1.1/.venv).
The zstd CLI (/usr/bin/zstd) produces interoperable frames - used for bulk
conversion of pre-existing shards.
"""
import io
import sys
from pathlib import Path

import numpy as np
from numcodecs import Zstd

_CODEC = Zstd(level=7)


def save_npy_zst(path, arr) -> None:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(_CODEC.encode(buf.getvalue()))
    tmp.replace(path)


def load_array(path):
    """*.npy.zst (or legacy *.npy) -> np.ndarray (float32, memmap-free)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    return np.load(io.BytesIO(_CODEC.decode(path.read_bytes())))


def save_npz_zst(path, **arrays) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(_CODEC.encode(buf.getvalue()))
    tmp.replace(path)


def load_npz(path):
    """*.npz.zst (or legacy *.npz) -> np.lib.npyio.NpzFile-like mapping."""
    path = Path(path)
    if path.suffix == ".npz":
        return np.load(path)
    return np.load(io.BytesIO(_CODEC.decode(path.read_bytes())))


def resolve(dirpath, stem, kind):
    """Preferred existing path for `<stem>.<kind>` (.zst first, legacy fallback)."""
    p = Path(dirpath)
    pz = p / f"{stem}.{kind}.zst"
    if pz.exists():
        return pz
    legacy = p / f"{stem}.{kind}"
    return legacy if legacy.exists() else pz


def _selftest():
    a = np.random.rand(90, 64, 64).astype(np.float32)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.npy.zst"
        save_npy_zst(f, a)
        assert np.array_equal(load_array(f), a)
        g = Path(td) / "t.npz.zst"
        save_npz_zst(g, surface=a[:4], level=a[4:])
        d = load_npz(g)
        assert np.array_equal(d["surface"], a[:4]) and np.array_equal(d["level"], a[4:])
    print("zstd_io selftest OK (npy + npz roundtrip lossless)")


if __name__ == "__main__":
    _selftest()
