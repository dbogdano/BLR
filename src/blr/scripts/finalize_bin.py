#!/usr/bin/env python3
"""
Finalize a single per-bin chunk file into a sorted FASTQ bin.

This script processes one chunk file (e.g. "ema-bin-000.chunk"), performs an
external merge-sort using memory-bounded runs, and writes the final FASTQ with
the same name but without the ".chunk" suffix (e.g. "ema-bin-000").

Intended to be launched as individual jobs (one per bin) from a job array or
parallel worker pool.

Usage:
  python3 src/blr/scripts/finalize_bin.py /path/to/ema-bin-000.chunk --max-lines 200000

"""
import argparse
import heapq
import os
from pathlib import Path
import tempfile
from typing import List, Iterator, Tuple


def parse_chunk_line(line: str) -> Tuple[int, List[str]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 7:
        return 0, parts
    try:
        heap_idx = int(parts[1])
    except Exception:
        heap_idx = 0
    return heap_idx, parts


def write_run(items: List[Tuple[int, List[str]]]) -> Path:
    run_fd, run_path = tempfile.mkstemp(prefix="run_", suffix=".tmp")
    os.close(run_fd)
    p = Path(run_path)
    with p.open("w") as fh:
        for heap_idx, parts in items:
            fh.write("\t".join([str(heap_idx)] + parts) + "\n")
    return p


def split_into_sorted_runs(path: Path, max_lines: int) -> List[Path]:
    runs = []
    with path.open("r") as fh:
        chunk = []
        for line in fh:
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


def iter_run(path: Path) -> Iterator[Tuple[int, List[str]]]:
    with path.open("r") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            try:
                heap_idx = int(parts[0])
            except Exception:
                heap_idx = 0
            yield heap_idx, parts[1:]


def merge_runs_and_write(runs: List[Path], out_path: Path):
    iterators = [iter_run(p) for p in runs]
    merged = heapq.merge(*iterators, key=lambda x: x[0])
    with out_path.open("w") as out:
        for heap_idx, parts in merged:
            if len(parts) < 6:
                continue
            name = parts[1]
            if not name.startswith("@"):
                header = "@" + name
            else:
                header = name
            seq1 = parts[2]
            qual1 = parts[3]
            seq2 = parts[4]
            qual2 = parts[5]
            out.write(f"{header}\n{seq1}\n+\n{qual1}\n")
            out.write(f"{header}\n{seq2}\n+\n{qual2}\n")


def finalize_chunk_file(chunk_path: Path, max_lines: int):
    final_path = chunk_path.with_suffix("")
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
    if chunk_path.stat().st_size == 0:
        final_path.open("w").close()
        try:
            chunk_path.unlink()
        except Exception:
            pass
        return str(final_path)
    runs = split_into_sorted_runs(chunk_path, max_lines=max_lines)
    try:
        merge_runs_and_write(runs, final_path)
    finally:
        for r in runs:
            try:
                r.unlink()
            except Exception:
                pass
        try:
            chunk_path.unlink()
        except Exception:
            pass
    return str(final_path)


def main():
    parser = argparse.ArgumentParser(description="Finalize one per-bin chunk file into a sorted FASTQ")
    parser.add_argument("chunk_file", help="Path to the per-bin chunk file to finalize (.chunk)")
    parser.add_argument("--max-lines", type=int, default=200000, help="Max lines to keep in memory per run")
    args = parser.parse_args()
    p = Path(args.chunk_file)
    if not p.exists():
        parser.error(f"chunk_file does not exist: {p}")
    print(finalize_chunk_file(p, args.max_lines))

if __name__ == '__main__':
    main()
