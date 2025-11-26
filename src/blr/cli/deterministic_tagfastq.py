"""
Create a deterministic barcode->bin mapping using greedy largest-first packing.

This command reads a starcode cluster file (canonical_seq, size, cluster_seqs)
and assigns canonical barcodes to `nr_bins` using a greedy algorithm that
places the largest clusters into the currently least-loaded bin.

Outputs:
 - mapping TSV: `canonical\tbin_index` (default `barcode_bins.tsv`)
 - summary CSV: `barcode_bins_summary.csv` with per-bin loads and counts

This mapping is deterministic and intended to be used by runtime tagging to
write reads directly into balanced per-bin FASTQ files.
"""
from pathlib import Path
import argparse
import heapq
import logging
import csv

logger = logging.getLogger(__name__)


def build_bin_mapping(clusters_file: str, nr_bins: int, chunksize: int = 100_000):
    """Build a canonical->bin mapping using greedy largest-first packing.

    Returns (mapping_dict, per_bin_reads, per_bin_counts)
    """
    items = []  # list of (canonical, size)
    # Import pandas lazily so that module import doesn't require pandas.
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to read the clusters file") from e

    cols = ["canonical_seq", "size", "cluster_seqs"]
    for chunk in pd.read_csv(clusters_file, sep="\t", names=cols,
                             dtype={"canonical_seq": str, "size": int, "cluster_seqs": str},
                             usecols=[0, 1], chunksize=chunksize):
        # drop missing
        chunk = chunk.dropna(subset=["canonical_seq"])
        for canonical, size in zip(chunk["canonical_seq"].tolist(), chunk["size"].tolist()):
            try:
                items.append((canonical, int(size)))
            except Exception:
                # skip malformed rows
                continue

    logger.info("Read %d canonical entries from clusters file", len(items))

    # Sort descending by size (largest-first)
    items.sort(key=lambda x: x[1], reverse=True)

    # Min-heap of (load, bin_index)
    heap = [(0, i) for i in range(nr_bins)]
    heapq.heapify(heap)

    mapping = {}
    per_bin_reads = [0] * nr_bins
    per_bin_counts = [0] * nr_bins

    for canonical, size in items:
        load, idx = heapq.heappop(heap)
        mapping[canonical] = idx
        load += size
        per_bin_reads[idx] += size
        per_bin_counts[idx] += 1
        heapq.heappush(heap, (load, idx))

    return mapping, per_bin_reads, per_bin_counts


def write_mapping_tsv(mapping, out_path: str):
    outp = Path(out_path)
    with outp.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["canonical_seq", "bin_index"])  # header
        for canonical, bin_idx in mapping.items():
            writer.writerow([canonical, bin_idx])


def write_summary(per_bin_reads, per_bin_counts, out_path: str):
    outp = Path(out_path)
    with outp.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bin_index", "total_reads", "num_canonicals"])
        for i, (reads, cnt) in enumerate(zip(per_bin_reads, per_bin_counts)):
            writer.writerow([i, reads, cnt])


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("clusters_file", help="Starcode clusters file (canonical\tsize\tcluster_seqs)")
    parser.add_argument("--nr-bins", type=int, default=1000, help="Number of bins to pack into")
    parser.add_argument("--chunksize", type=int, default=100000, help="Pandas chunksize when reading clusters file")
    parser.add_argument("--out-mapping", default="barcode_bins.tsv", help="Output mapping TSV path")
    parser.add_argument("--out-summary", default="barcode_bins_summary.csv", help="Output per-bin summary CSV path")


def main(args):
    clusters_file = args.clusters_file
    nr_bins = args.nr_bins
    chunksize = args.chunksize
    out_mapping = args.out_mapping
    out_summary = args.out_summary

    logger.info("Building deterministic bin mapping for %s into %d bins", clusters_file, nr_bins)
    mapping, per_bin_reads, per_bin_counts = build_bin_mapping(clusters_file, nr_bins, chunksize=chunksize)

    logger.info("Writing mapping to %s (this may be large)", out_mapping)
    write_mapping_tsv(mapping, out_mapping)

    logger.info("Writing per-bin summary to %s", out_summary)
    write_summary(per_bin_reads, per_bin_counts, out_summary)

    max_reads = max(per_bin_reads) if per_bin_reads else 0
    min_reads = min(per_bin_reads) if per_bin_reads else 0
    avg_reads = sum(per_bin_reads) / len(per_bin_reads) if per_bin_reads else 0
    logger.info("Per-bin reads: min=%d max=%d avg=%.2f", min_reads, max_reads, avg_reads)
