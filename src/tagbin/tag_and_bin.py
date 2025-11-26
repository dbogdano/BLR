"""Tag FASTQ reads and write them into per-bin FASTQ files using a provided mapping.

This compact script uses `dnaio` for FASTQ I/O and reads a tab-delimited
`canonical_seq\tbin_index` mapping produced by the bin builder.

All heavy imports are performed lazily inside functions so the module can be
imported without installing dependencies (good for quick checks).
"""
from pathlib import Path
import argparse
import csv
import logging

logger = logging.getLogger(__name__)


def load_bin_map(tsv_path: str):
    mapping = {}
    with open(tsv_path, newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for row in rdr:
            try:
                mapping[row['canonical_seq']] = int(row['bin_index'])
            except Exception:
                continue
    return mapping


def tag_and_bin(uncorrected_barcodes: str, corrected_clusters: str, reads1: str, reads2: str or None,
                out_dir: str, bin_map: str, mapper: str = "ema", nr_bins: int = 100):
    # lazy imports
    try:
        import dnaio
    except Exception as e:
        raise RuntimeError("dnaio is required to run tag-and-bin") from e
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to read clusters (for building mapping) or for lookup") from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load or build canonical mapping
    if bin_map is None:
        # build a simple raw->canonical map from clusters (not memory optimal but compact)
        df = pd.read_csv(corrected_clusters, sep="\t", names=["canonical_seq", "size", "cluster_seqs"], dtype={"canonical_seq": str, "cluster_seqs": str})
        corrected = {s: canonical for canonical, seqs in zip(df["canonical_seq"], df["cluster_seqs"]) for s in seqs.split(",")}
        bin_map_local = None
    else:
        bin_map_local = load_bin_map(bin_map)
        corrected = None

    # Open barcode reader and reads
    barcode_reader = dnaio.open(uncorrected_barcodes, mode="r")
    barcodes_iter = iter(barcode_reader)

    # prepare per-bin writers lazily
    bin_writers = {}

    def get_bin_writer(idx):
        if idx not in bin_writers:
            fname = out_dir / f"ema-bin-{str(idx).zfill(3)}"
            bin_writers[idx] = dnaio.open(str(fname), interleaved=True, mode="w", fileformat="fastq")
        return bin_writers[idx]

    # helper to get barcode for a read header
    def get_raw_bc(read_name):
        # read_name is expected to be the first token (no spaces)
        # advance barcodes iterator until we find matching header
        nonlocal barcodes_iter
        cache = {}
        for barcode in barcodes_iter:
            bid = barcode.name.partition(" ")[0]
            cache[bid] = barcode.sequence
            if bid == read_name:
                return cache[bid]
        return None

    # Open reads iterator
    reads = dnaio.open(reads1, file2=reads2, interleaved=(reads2 is None), mode="r")

    for r1, r2 in reads:
        name = r1.name.split(maxsplit=1)[0]
        raw_bc = get_raw_bc(name)
        if raw_bc is None:
            continue

        # find canonical
        canonical = None
        if corrected is not None:
            canonical = corrected.get(raw_bc)
        else:
            # if we have only bin_map, we need a way to get canonical -> but bin_map keys are canonical
            # So we attempt to use raw_bc directly; if not found, skip
            canonical = raw_bc if raw_bc in bin_map_local else None

        if canonical is None:
            continue

        # determine bin
        if bin_map_local is not None:
            bin_idx = bin_map_local.get(canonical)
            if bin_idx is None:
                # fallback hash using provided nr_bins
                bin_idx = hash(canonical) % nr_bins
        else:
            bin_idx = hash(canonical) % nr_bins

        w = get_bin_writer(bin_idx)
        # write interleaved
        w.write(r1, r2)

    # close writers
    reads.close()
    barcode_reader.close()
    for w in bin_writers.values():
        w.close()


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("uncorrected_barcodes", help="FASTQ of raw barcodes")
    parser.add_argument("corrected_clusters", help="starcode clusters TSV")
    parser.add_argument("reads1", help="reads R1 FASTQ")
    parser.add_argument("reads2", nargs='?', default=None, help="reads R2 FASTQ (optional for paired)")
    parser.add_argument("--out-dir", required=True, help="Directory to write per-bin FASTQ")
    parser.add_argument("--bin-map", default=None, help="Optional canonical->bin TSV produced by bin_builder")
    parser.add_argument("--nr-bins", type=int, default=100, help="Number of bins to use for hashing fallback and writer setup")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tag reads and write to bins")
    add_arguments(parser)
    args = parser.parse_args(argv)
    tag_and_bin(args.uncorrected_barcodes, args.corrected_clusters, args.reads1, args.reads2, args.out_dir, args.bin_map, nr_bins=args.nr_bins)


if __name__ == "__main__":
    main()
