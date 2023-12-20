import glob
import gzip
import itertools
import json
import lzma
import os
import sqlite3
from typing import Any, Callable, Dict, Iterable, Iterator, List, Union

import numpy as np

try:
    import zstandard
except ImportError:
    zstandard = None

try:
    import orjson
except ImportError:
    orjson = json


class ReservoirSampler:
    def __init__(self, size, rng=None):
        self.size = size
        self.rng = rng or np.random.default_rng()

        self.reservoir = []
        self.n_seen = 0

    def add(self, x):
        self.n_seen += 1
        if len(self.reservoir) < self.size:
            self.reservoir.append(x)
        else:
            idx = self.rng.integers(0, self.n_seen)
            if idx < self.size:
                self.reservoir[idx] = x

    def add_many(self, xs):
        for x in xs:
            self.add(x)

    def get_reservoir(self):
        return self.reservoir


def openc(fname, mode="rt", *, compression="auto", **kwargs):
    """Opens a file, transparently handling a variety of possible compression schemes."""
    if mode == "w":
        mode = "wt"

    if mode == "x":
        mode = "xt"

    kwargs["mode"] = mode
    if compression == "auto":
        # TODO: Maybe use magic number instead of extension
        if fname.lower().endswith(".gz"):
            compression = "gzip"
        elif fname.lower().endswith(".xz"):
            compression = "lzma"
        elif fname.lower().endswith(".zst"):
            compression = "zstd"
        else:
            compression = "none"

    open_fn = open
    if compression == "gzip":
        open_fn = gzip.open
    elif compression == "lzma":
        open_fn = lzma.open
    elif compression == "zstd":
        if zstandard is None:
            raise ValueError("zstandard module is not available")
        open_fn = zstandard.open

    return open_fn(fname, **kwargs)


def iter_jsonl_files(infiles):
    if isinstance(infiles, str):
        infiles = [infiles]
    for infile in infiles:
        with openc(infile) as f:
            for obj in map(orjson.loads, f):
                yield obj


def zip_strict(*iterables):
    # Until python 3.10, seems like there's no builtin way to do this, but
    # there's a fairly simple workaround implementation:
    # https://stackoverflow.com/a/32954700
    canary = object()
    for tup in itertools.zip_longest(*iterables, fillvalue=canary):
        if canary in tup:
            raise ValueError("Iterables have different lengths")
        yield tup


def downsample_recs(recs: List[Any], downsample_config: Dict[str, Any]):
    if downsample_config is None:
        # Return recs, for consistency with old configs before downsampling was added
        return recs.copy()

    if downsample_config.get("keep_n", -1) != -1 and downsample_config.get("keep_ratio", -1) != -1:
        raise ValueError("Need only one of keep_n and keep_ratio (not both)")

    keep_n = len(recs)
    if "keep_n" in downsample_config:
        keep_n = downsample_config["keep_n"]
    elif "keep_ratio" in downsample_config:
        keep_n = max(1, int(downsample_config["keep_ratio"] * len(recs)))

    assert isinstance(keep_n, int) and keep_n > 0

    if keep_n > len(recs):
        raise ValueError("Can't sample more data points than the dataset has")

    rng = np.random.default_rng(downsample_config.get("seed", None))
    return [recs[idx] for idx in rng.choice(len(recs), size=keep_n, replace=False)]


def batch_iter(iterable, batch_size):
    batch = []
    for rec in iterable:
        if len(batch) >= batch_size:
            yield batch
            batch = []
        batch.append(rec)

    if len(batch) != 0:
        yield batch


def index_by(
    lst: Union[Iterable, Iterator],
    key: Union[str, Callable],
    one_to_one=False,
) -> Dict:
    key_fn = key
    if isinstance(key_fn, str):
        key_fn = lambda x: x[key]

    index = dict()
    if one_to_one:
        for rec in lst:
            k = key_fn(rec)
            if k in index:
                raise ValueError("Duplicate key: {}".format(k))
            index[k] = rec
    else:
        for rec in lst:
            k = key_fn(rec)
            if k not in index:
                index[k] = []
            index[k].append(rec)
    return index


def deduplicate_by(
    lst: Union[Iterable, Iterator],
    key: Union[str, Callable],
) -> List:
    key_fn = key
    if isinstance(key_fn, str):
        key_fn = lambda x: x[key]

    new_lst = []
    used_keys = set()
    for rec in lst:
        k = key_fn(rec)
        if k not in used_keys:
            used_keys.add(k)
            new_lst.append(rec)
    return new_lst


def counter_jaccard(counter1: Dict, counter2: Dict) -> float:
    """Computes the jaccard overlap of two dict objects."""
    if len(counter1) == 0 and len(counter2) == 0:
        return float("nan")
    if len(counter1) == 0 or len(counter2) == 0:
        return 0.0

    intersection = sum((counter1 & counter2).values())
    if intersection == 0:
        return 0.0
    return intersection / (sum(counter1.values()) + sum(counter2.values()) - intersection)
