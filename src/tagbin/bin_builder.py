"""Greedy largest-first bin packing builder.

Reads a starcode-style clusters TSV (canonical_seq\tsize\tcluster_seqs)
and assigns canonical sequences to bins using a greedy min-heap packer.

This module imports `pandas` lazily to avoid hard dependency at module import time.
"""
from pathlib import Path
import argparse
import heapq
import csv
import logging

logger = logging.getLogger(__name__)


def build_bin_mapping(clusters_file: str, nr_bins: int, chunksize: int = 100_000):
    """Read clusters_file and return (mapping_dict, per_bin_reads, per_bin_counts).

    clusters_file: path to TSV with columns: canonical_seq, size, cluster_seqs
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required for reading clusters file") from e

    cols = ["canonical_seq", "size", "cluster_seqs"]
    items = []
    for chunk in pd.read_csv(clusters_file, sep="\t", names=cols, usecols=[0, 1],
                             dtype={"canonical_seq": str, "size": int}, chunksize=chunksize):
        chunk = chunk.dropna(subset=["canonical_seq"])
        items.extend(list(zip(chunk["canonical_seq"].tolist(), chunk["size"].tolist())))

    logger.info("Loaded %d canonical entries", len(items))

    # sort descending by size
    items.sort(key=lambda x: x[1], reverse=True)

    # min-heap of (load, bin_index)
    heap = [(0, i) for i in range(nr_bins)]
    heapq.heapify(heap)

    mapping = {}
    per_bin_reads = [0] * nr_bins
    per_bin_counts = [0] * nr_bins

    for canonical, size in items:
        load, idx = heapq.heappop(heap)
        mapping[canonical] = idx
        load += int(size)
        per_bin_reads[idx] += int(size)
        per_bin_counts[idx] += 1
        heapq.heappush(heap, (load, idx))

    return mapping, per_bin_reads, per_bin_counts


def write_mapping(mapping, out_tsv: str):
    out = Path(out_tsv)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["canonical_seq", "bin_index"])
        for k, v in mapping.items():
            writer.writerow([k, v])


def write_summary(per_bin_reads, per_bin_counts, out_csv: str):
    out = Path(out_csv)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bin_index", "total_reads", "num_canonicals"])
        for i, (reads, cnt) in enumerate(zip(per_bin_reads, per_bin_counts)):
            writer.writerow([i, reads, cnt])


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("clusters_file", help="Starcode clusters TSV (canonical\tsize\tcluster_seqs)")
    parser.add_argument("--nr-bins", type=int, default=100, help="Number of bins to create")
    parser.add_argument("--out-mapping", default="barcode_bins.tsv", help="Output mapping TSV")
    parser.add_argument("--out-summary", default="barcode_bins_summary.csv", help="Per-bin summary CSV")
    parser.add_argument("--chunksize", type=int, default=100000, help="Pandas chunksize")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Greedy canonical->bin mapper")
    add_arguments(parser)
    args = parser.parse_args(argv)

    mapping, per_bin_reads, per_bin_counts = build_bin_mapping(args.clusters_file, args.nr_bins, chunksize=args.chunksize)
    write_mapping(mapping, args.out_mapping)
    write_summary(per_bin_reads, per_bin_counts, args.out_summary)

    total = sum(per_bin_reads)
    logger.info("Done. total reads assigned=%d bins=%d", total, args.nr_bins)


if __name__ == "__main__":
    main()
