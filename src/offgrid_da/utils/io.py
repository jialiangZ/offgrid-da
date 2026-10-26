"""HTTP-based object fetching with a local file cache.

The weatherbench2 GCS bucket is public, so objects can be fetched directly
via ``https://storage.googleapis.com/<bucket>/<object>``. A disk cache avoids
repeated downloads during development (plan section 3.10).

Note: we deliberately read METAR parquet over plain HTTPS instead of gcsfs
because the installed gcsfs 2026.10.1 mangles non-zarr object paths
(verified in Phase 0); zarr datasets are read through xarray + gcsfs which
works correctly.
"""
from __future__ import annotations

import hashlib
import io
import time
import urllib.error
import urllib.request
from pathlib import Path

GCS_HTTP_BASE = "https://storage.googleapis.com"

_RETRIES = 4
_RETRY_BACKOFF_S = 2.0


def gs_to_https(url: str) -> str:
    """Convert ``gs://bucket/path`` to the public HTTPS URL."""
    if url.startswith("gs://"):
        return f"{GCS_HTTP_BASE}/{url[len('gs://'):]}"
    return url


def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    """Fetch ``url`` (gs:// or https://) with retries, returning bytes."""
    https_url = gs_to_https(url)
    last_err: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(https_url, headers={"User-Agent": "offgrid-da/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as err:  # noqa: PERF203
            last_err = err
            time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {https_url} after {_RETRIES} attempts: {last_err}")


def cached_fetch(url: str, cache_dir: str | Path) -> bytes:
    """Fetch with a content-addressed local cache under ``cache_dir``."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(gs_to_https(url).encode()).hexdigest()
    cached_file = cache_path / key
    if cached_file.exists():
        return cached_file.read_bytes()
    data = fetch_bytes(url)
    tmp = cached_file.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(cached_file)
    return data


def read_parquet_cached(url: str, cache_dir: str | Path):
    """Read a parquet object as a pandas DataFrame with local caching."""
    import pandas as pd

    return pd.read_parquet(io.BytesIO(cached_fetch(url, cache_dir)))
