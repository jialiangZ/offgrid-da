#!/usr/bin/env python3
"""Compression benchmark for AIFS pipeline data (small batch, per user request).

Samples: 3 input shards (90ch x 0.25deg, Jan/Feb/Apr) + 1 regenerated background.
Candidates: zstd level sweep, lz4, xz, and shuffle variants (blosc zstd + bit/byte
shuffle) - shuffling byte planes often boosts float32 arrays substantially.

Reports: ratio, encode time, decode speed, lossless check.
Run with repo .venv (numcodecs). Heavy codecs (zstd-19, xz) run on ONE shard only.
"""
import io
import subprocess
import time
from pathlib import Path

import numpy as np
from numcodecs import Zstd, LZ4, Blosc

HERE = Path(__file__).resolve().parent
SHARDS = [HERE / "data/aifs_inputs/2021" / f"{s}.npy" for s in
          ("2021010106", "2021021512", "2021043018")]
BG = sorted((HERE / "data/bg_aifs_2021").glob("state_*.npz"))
GIB = 2**30


def bench_numpy_codec(name, codec, arr):
    raw = arr.tobytes()
    t0 = time.perf_counter()
    enc = codec.encode(raw)
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    dec = codec.decode(enc)
    t_dec = time.perf_counter() - t0
    ok = dec == raw
    return name, len(enc) / len(raw), t_enc, len(raw) / t_dec / GIB, ok


def bench_cli(name, args, raw_path, raw_len):
    t0 = time.perf_counter()
    subprocess.run(args, check=True, capture_output=True)
    t_enc = time.perf_counter() - t0
    out = Path(str(raw_path) + ".bench.zst" if "zstd" in args[0] else str(raw_path) + ".bench.xz")
    t0 = time.perf_counter()
    subprocess.run(["zstd", "-dc", str(out)] if out.suffix == ".zst" else ["xz", "-dc", str(out)],
                   check=True, capture_output=True)
    t_dec = time.perf_counter() - t0
    out.unlink()
    return name, out.stat().st_size / raw_len, t_enc, raw_len / t_dec / GIB, None


def main():
    arrays = [np.load(p) for p in SHARDS]
    print(f"samples: {[p.name for p in SHARDS]}, each {arrays[0].nbytes/GIB:.3f} GiB raw\n")

    configs = [
        ("zstd-1", lambda: Zstd(level=1)),
        ("zstd-3", lambda: Zstd(level=3)),
        ("zstd-7", lambda: Zstd(level=7)),
        ("zstd-12", lambda: Zstd(level=12)),
        ("lz4", lambda: LZ4()),
        ("blosc-zstd7-noshuf", lambda: Blosc(cname="zstd", clevel=7, shuffle=Blosc.NOSHUFFLE)),
        ("blosc-zstd7-byteshuf", lambda: Blosc(cname="zstd", clevel=7, shuffle=Blosc.SHUFFLE)),
        ("blosc-zstd7-bitshuf", lambda: Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE)),
        ("blosc-zstd9-noshuf", lambda: Blosc(cname="zstd", clevel=9, shuffle=Blosc.NOSHUFFLE, blocksize=2**28)),
        ("blosc-zstd9-byteshuf", lambda: Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE, blocksize=2**28)),
    ]
    print(f"{'codec':>22} {'ratio':>7} {'enc s':>7} {'dec GiB/s':>9} {'lossless':>8}")
    results = {}
    for name, mk in configs:
        row = []
        for i, arr in enumerate(arrays):
            r = bench_numpy_codec(name, mk(), arr)
            row.append(r)
            if i == 0:  # full detail for shard 1
                print(f"{r[0]:>22} {r[1]:>7.3f} {r[2]:>7.1f} {r[3]:>9.2f} {str(r[4]):>8}")
        results[name] = [r[1] for r in row]
    print("ratios per shard:", {k: [round(x, 3) for x in v] for k, v in results.items()})

    # heavy codecs on shard 1 only, via CLI
    print("\nheavy codecs (shard 1 only):")
    raw_path = SHARDS[0]
    n = arrays[0].nbytes
    outs = {"zstd-7-long27": raw_path.with_suffix(".npy.bench.zst"),
            "zstd-19": raw_path.with_suffix(".npy.bench.zst"),
            "zstd-19-long27": raw_path.with_suffix(".npy.bench.zst"),
            "xz-6": Path(str(raw_path) + ".xz")}
    for name, args in [("zstd-7-long27", ["zstd", "-7", "--long=27", "-q", "-f", str(raw_path), "-o", str(outs["zstd-19"])]),
                       ("zstd-19", ["zstd", "-19", "-q", "-f", str(raw_path), "-o", str(outs["zstd-19"])]),
                       ("zstd-19-long27", ["zstd", "-19", "--long=27", "-q", "-f", str(raw_path), "-o", str(outs["zstd-19-long27"])]),
                       ("xz-6", ["xz", "-6", "-q", "-k", "-f", "-T", "8", str(raw_path)])]:
        out = outs[name]
        t0 = time.perf_counter()
        subprocess.run(args, check=True, capture_output=True)
        t_enc = time.perf_counter() - t0
        dec_args = ["zstd", "-dc", str(out)] if out.name.endswith(".zst") else ["xz", "-dc", str(out)]
        t0 = time.perf_counter()
        subprocess.run(dec_args, check=True, capture_output=True)
        t_dec = time.perf_counter() - t0
        print(f"{name:>22} {out.stat().st_size / n:>7.3f} {t_enc:>7.1f} {n / t_dec / GIB:>9.2f}")
        out.unlink(missing_ok=True)

    # background npz comparison
    print("\nbackground npz (one regenerated box-avg file):")
    if BG:
        f = BG[0]
        d = np.load(f)
        arrays_npz = {k: d[k] for k in d.files}
        buf = io.BytesIO()
        np.savez(buf, **arrays_npz)
        raw_npz = buf.getvalue()
        zlib_size = f.stat().st_size
        print(f"raw members {len(raw_npz)/2**20:.1f} MiB | current zlib npz {zlib_size/2**20:.1f} MiB "
              f"({zlib_size/len(raw_npz):.3f}x)")
        best = Blosc(cname="zstd", clevel=7, shuffle=Blosc.SHUFFLE)
        # per-array blosc on concatenated physical values
        cat = np.concatenate([arrays_npz['surface'].ravel(), arrays_npz['level'].ravel()])
        for nm, cd in [("plain-zstd7", Zstd(level=7)),
                       ("cat-byteshuf-zstd7", Blosc(cname="zstd", clevel=7, shuffle=Blosc.SHUFFLE)),
                       ("cat-bitshuf-zstd7", Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE))]:
            enc = cd.encode(cat.tobytes())
            print(f"{nm:>20}: {len(enc)/2**20:7.2f} MiB ({len(enc)/cat.nbytes:.3f}x)")


if __name__ == "__main__":
    main()
