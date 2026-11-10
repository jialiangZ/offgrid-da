import torch

from offgrid_da.data.arches_training import LazyArchesDataset, StationMembership
from offgrid_da.data.metar import MetarReader
from offgrid_da.data.stations import station_role


def test_lazy_constructor_does_not_scan_observations(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Construction scanned observations")
    monkeypatch.setattr(MetarReader, "read_window", forbidden)
    ds = LazyArchesDataset("/home/jlzhang/Documents/offgrid-da", "train", cache_root=tmp_path)
    assert len(ds) == 4384
    assert not list(tmp_path.rglob("*.pt"))


def test_lazy_cache_uses_identical_normalized_sample(monkeypatch, tmp_path):
    ds = LazyArchesDataset("/home/jlzhang/Documents/offgrid-da", "train", cache_root=tmp_path)
    first = ds[0]
    def forbidden(*args, **kwargs):
        raise AssertionError("Cache hit queried observations")
    monkeypatch.setattr(MetarReader, "read_window", forbidden)
    again = ds[0]
    torch.testing.assert_close(first["background_normalized"], again["background_normalized"], rtol=0, atol=0)
    assert first["context_station_ids"] == again["context_station_ids"]
    ds.validate_sample(again)


def test_hashed_membership_is_independent_of_time_window():
    context = StationMembership("context", "default", .2, 42)
    heldout = StationMembership("heldout", "default", .2, 42)
    for station in ("VHHH", "KJFK", "EGLL", "RJTT"):
        assert (station in context) == (station_role(station) == "context")
        assert (station in context) != (station in heldout)

