#!/usr/bin/env python3
"""
Finalize per-bin chunk files into sorted FASTQ bins.

This script reads files matching a pattern (default: "ema-bin-*.chunk") in a bins
directory, sorts each file by the numeric heap index (2nd tab column) using an
external merge sort (memory-limited), and writes final interleaved FASTQ files
named like `ema-bin-000`, `ema-bin-001`, ...

Each chunk line format is expected to be:
  canonical\theap\tname\tseq1\tqual1\tseq2\tqual2\n

Usage examples:
  python3 src/blr/scripts/finalize_bins.py /path/to/bins_dir
  python3 src/blr/scripts/finalize_bins.py /path/to/bins_dir --pattern "ema-bin-*.chunk" --max-lines 200000

"""
import argparse
import heapq
import os
from pathlib import Path
import tempfile
from typing import List, Iterator, Tuple
import concurrent.futures


def parse_chunk_line(line: str) -> Tuple[int, List[str]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 7:
        return 0, parts
    try:
        heap_idx = int(parts[1])
    except Exception:
        heap_idx = 0
    return heap_idx, parts


def split_into_sorted_runs(path: Path, max_lines: int) -> List[Path]:
    """Read the chunk file and produce sorted run files, each with at most max_lines lines."""
    runs = []
    with path.open("r") as fh:
        chunk = []
        for i, line in enumerate(fh, 1):
            heap_idx, parts = parse_chunk_line(line)
            chunk.append((heap_idx, parts))
            if len(chunk) >= max_lines:
                chunk.sort(key=lambda x: x[0])
                run_path = write_run(chunk)
                runs.append(run_path)
                chunk = []
        if chunk:
            chunk.sort(key=lambda x: x[0])
            run_path = write_run(chunk)
            runs.append(run_path)
    return runs


def write_run(items: List[Tuple[int, List[str]]]) -> Path:
    run_fd, run_path = tempfile.mkstemp(prefix="run_", suffix=".tmp")
    os.close(run_fd)
    p = Path(run_path)
    with p.open("w") as fh:
        for heap_idx, parts in items:
            fh.write("\t".join([str(heap_idx)] + parts) + "\n")
    return p


def iter_run(path: Path) -> Iterator[Tuple[int, List[str]]]:
    with path.open("r") as fh:
        for line in fh:
            # line is heap \t rest..., we keep the original parts (with canonical repeated)
            parts = line.rstrip("\n").split("\t")
            try:
                heap_idx = int(parts[0])
            except Exception:
                heap_idx = 0
            # parts[1:] corresponds to original parts
            yield heap_idx, parts[1:]


def merge_runs_and_write(runs: List[Path], out_path: Path):
    """Merge sorted run files and write final interleaved FASTQ to out_path."""
    # Create generators for each run
    iterators = [iter_run(p) for p in runs]
    # Use heapq.merge to merge by heap index
    merged = heapq.merge(*iterators, key=lambda x: x[0])
    # Write final FASTQ
    with out_path.open("w") as out:
        for heap_idx, parts in merged:
            # parts: [canonical, name, seq1, qual1, seq2, qual2] OR may include canonical twice
            if len(parts) < 6:
                continue
            # Name may already include leading @; ensure FASTQ header starts with '@'
            name = parts[1]
            if not name.startswith("@"):
                header = "@" + name
            else:
                header = name
            seq1 = parts[2]
            qual1 = parts[3]
            seq2 = parts[4]
            qual2 = parts[5]
            # Write read1
            out.write(f"{header}\n{seq1}\n+\n{qual1}\n")
            # Write read2 (name re-used)
            out.write(f"{header}\n{seq2}\n+\n{qual2}\n")


def finalize_chunk_file(chunk_path: Path, max_lines: int):
    # Determine final path (remove .chunk suffix)
    final_path = chunk_path.with_suffix("")
    # If there are no lines, just touch the final file and remove chunk
    if chunk_path.stat().st_size == 0:
        final_path.open("w").close()
        try:
            chunk_path.unlink()
        except Exception:
            pass
        return str(final_path)

    # Split into runs
    runs = split_into_sorted_runs(chunk_path, max_lines=max_lines)
    try:
        if len(runs) == 1:
            # Single sorted run; rename to temp and then write to final by reading
            # We still need to write final in FASTQ format
            merge_runs_and_write(runs, final_path)
        else:
            merge_runs_and_write(runs, final_path)
    finally:
        # Cleanup run files
        for r in runs:
            try:
                r.unlink()
            except Exception:
                pass
        # Remove original chunk
        try:
            chunk_path.unlink()
        except Exception:
            pass
    return str(final_path)


def find_chunk_files(bins_dir: Path, pattern: str) -> List[Path]:
    return sorted(bins_dir.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Finalize per-bin chunk files into sorted FASTQ bins")
    parser.add_argument("bins_dir", help="Directory containing per-bin chunk files")
    parser.add_argument("--pattern", default="ema-bin-*.chunk", help="Glob pattern for chunk files")
    parser.add_argument("--max-lines", type=int, default=200000, help="Max lines to keep in memory per run")
    parser.add_argument("--workers", type=int, default=1, help="Number of bins to process in parallel")
    args = parser.parse_args()

    bins_dir = Path(args.bins_dir)
    if not bins_dir.is_dir():
        parser.error(f"bins_dir does not exist: {bins_dir}")

    chunk_files = find_chunk_files(bins_dir, args.pattern)
    if not chunk_files:
        print("No chunk files found")
        return

    if args.workers and args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(finalize_chunk_file, p, args.max_lines) for p in chunk_files]
            for f in concurrent.futures.as_completed(futures):
                try:
                    print("finalized:", f.result())
                except Exception as e:
                    print("failed:", e)
    else:
        for p in chunk_files:
            print(finalize_chunk_file(p, args.max_lines))


if __name__ == '__main__':
    main()
