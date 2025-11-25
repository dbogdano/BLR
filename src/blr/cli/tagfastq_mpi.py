"""
Tag FASTQ headers with barcodes using MPI for distributed processing.

This version of tagfastq.py is modified to use MPI for parallel processing on HPC clusters.
Requires mpi4py to be installed in the environment.
"""
from contextlib import ExitStack
from heapq import merge
from itertools import islice, cycle
import logging
from pathlib import Path
import os
import sys
import tempfile
from typing import Dict, List, Tuple, Optional
import math

import dnaio
import pandas as pd
from xopen import xopen
from mpi4py import MPI
import hashlib

# Reuse helper classes/functions from serial tagfastq implementation
from blr.cli.tagfastq import BarcodeReader, map_corrected_barcodes, Output
from blr.cli._barcode_db import build_barcode_lmdb, build_barcode_sqlite, open_sqlite_readonly, open_lmdb_readonly, lookup_canonical, lookup_lmdb
import sqlite3

from blr.utils import tqdm, Summary, ACCEPTED_READ_MAPPERS

logger = logging.getLogger(__name__)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "M": "AC", "K": "GT",
    "S": "CG", "W": "AT", "H": "ACT", "B": "CGT",
    "V": "ACG", "D": "AGT", "N": "ACGT"
}

def distribute_workload(total_reads: int) -> Tuple[int, int]:
    """
    Calculate the chunk of reads this MPI rank should process.
    
    Args:
        total_reads: Total number of reads to process
        
    Returns:
        Tuple of (start_idx, end_idx) for this rank's chunk
    """
    chunk_size = math.ceil(total_reads / size)
    start_idx = rank * chunk_size
    end_idx = min((rank + 1) * chunk_size, total_reads)
    return start_idx, end_idx

def count_reads(input1: str, input2: str) -> int:
    """Count total number of reads in input files"""
    count = 0
    with dnaio.open(input1, file2=input2, mode="r") as reader:
        for _ in reader:
            count += 1
    return count


def barcode_to_bin(barcode: str, nr_bins: int) -> int:
    """Deterministically map a barcode string to a bin index using MD5.

    Uses the first 8 bytes of MD5 digest as an integer and reduces modulo nr_bins.
    """
    if nr_bins is None or nr_bins <= 0:
        return 0
    md5 = hashlib.md5(barcode.encode("ascii") if isinstance(barcode, str) else barcode)
    # use first 8 bytes to avoid huge integers but still be well distributed
    val = int.from_bytes(md5.digest()[:8], "big")
    return val % nr_bins


# Barcode DB helper functions (LMDB/SQLite builders and readers) live in
# `blr.cli.barcode_db` and are imported at the top of this file. This avoids
# circular imports and centralizes disk-backed barcode lookup logic.

def main(args):
    # Only rank 0 prints the start message
    if rank == 0:
        logger.info("Starting MPI version with %d processes", size)
    
    run_tagfastq_mpi(
        uncorrected_barcodes=args.uncorrected_barcodes,
        corrected_barcodes=args.corrected_barcodes,
        input1=args.input1,
        input2=args.input2,
        output1=args.output1,
        output2=args.output2,
        output_nobc1=args.output_nobc1,
        output_nobc2=args.output_nobc2,
        output_bins=args.output_bins,
        nr_bins=args.nr_bins,
        tmpdir=args.tmpdir,
        barcode_tag=args.barcode_tag,
        sequence_tag=args.sequence_tag,
        mapper=args.mapper,
        min_count=args.min_count,
        pattern_match=args.pattern_match,
        sample_number=args.sample_nr,
        build_db=args.build_db if hasattr(args, 'build_db') else False,
        lmdb_map_size=getattr(args, 'lmdb_map_size', None),
    )

def run_tagfastq_mpi(
        uncorrected_barcodes: str,
        corrected_barcodes: str,
        input1: str,
        input2: str,
        output1: str,
        output2: str,
        output_nobc1: str,
        output_nobc2: str,
        output_bins: str,
        nr_bins: int,
        tmpdir: str, 
        barcode_tag: str,
        sequence_tag: str,
        mapper: str,
        min_count: int,
        pattern_match: str,
        sample_number: int,
        build_db: bool = False,
        lmdb_map_size: Optional[int] = None,
):
    summary = Summary()
    
    # Optionally, only build DB and exit: build on rank 0 then barrier and return
    if build_db:
        if rank == 0:
            logger.info("Building LMDB barcode DB (build_db requested)")
            template = [set(IUPAC[base]) for base in pattern_match] if pattern_match else []
            db_path = str(Path(tmpdir) / "barcode_mapping.lmdb")
            map_size = lmdb_map_size if lmdb_map_size is not None else (1 << 34)
            build_barcode_lmdb(corrected_barcodes, db_path, summary, mapper, template, min_count, chunksize=10000, map_size=map_size)
            logger.info(f"Barcode DB written to {db_path}")
        # synchronize and exit
        comm.Barrier()
        if rank == 0:
            summary.print_stats(__name__)
            logger.info("Finished building DB")
        return

    # Only rank 0 builds the barcode LMDB and broadcasts the DB path and heap
    if rank == 0:
        logger.info("Map clusters and build barcode DB (LMDB)")
        template = [set(IUPAC[base]) for base in pattern_match] if pattern_match else []
        db_path = str(Path(tmpdir) / "barcode_mapping.lmdb")
        # build_barcode_lmdb returns (db_path, heap_index)
        db_path, heap = build_barcode_lmdb(corrected_barcodes, db_path, summary, mapper, template, min_count)
    else:
        db_path = None
        heap = None

    # Broadcast the barcode DB path and heap to all processes
    db_path = comm.bcast(db_path, root=0)
    heap = comm.bcast(heap, root=0)

    # Count total reads (only rank 0)
    if rank == 0:
        total_reads = count_reads(input1, input2)
    else:
        total_reads = None
    
    # Broadcast total reads count to all processes
    total_reads = comm.bcast(total_reads, root=0)
    
    # Calculate this rank's portion of reads to process
    start_idx, end_idx = distribute_workload(total_reads)
    
    # Setup output handling based on rank
    if rank == 0:
        output_handler = setup_output(output1, output2, output_nobc1, output_nobc2, 
                                    output_bins, mapper, nr_bins)
    else:
        # Non-root processes write to temporary files
        tmp_output = f"{tmpdir}/rank_{rank}_output.fastq"
        output_handler = setup_output(tmp_output, None, None, None, None, mapper, nr_bins)

    # Process reads for this rank's chunk
    process_reads_chunk(
        input1, input2, start_idx, end_idx, db_path, heap,
        output_handler, barcode_tag, sequence_tag, mapper, summary,
        uncorrected_barcodes=uncorrected_barcodes, output_bins=output_bins, nr_bins=nr_bins, tmpdir=tmpdir, rank=rank
    )
    
    # Synchronize all processes
    comm.Barrier()
    
    if rank == 0:
        # Merge results from all ranks
        if output_bins is not None:
            merge_rank_outputs(tmpdir, output_bins, size, nr_bins)
        else:
            merge_rank_outputs(tmpdir, output1, size, None)

        # Print final summary
        summary.print_stats(__name__)
        logger.info("Finished")

def process_reads_chunk(input1, input2, start_idx, end_idx, barcode_db_path, heap,
                       output_handler, barcode_tag, sequence_tag, mapper, summary,
                       uncorrected_barcodes=None, output_bins=None, nr_bins=None, tmpdir=".", rank=0):
    """Process a chunk of reads assigned to this rank.

    For EMA + output_bins we deterministically partition reads by corrected barcode using
    MD5 hashing and write per-bin, per-rank files which are merged later.
    """
    with ExitStack() as stack:
        reader = stack.enter_context(dnaio.open(input1, file2=input2, mode="r"))
        uncorrected_barcode_reader = stack.enter_context(BarcodeReader(uncorrected_barcodes))

        # Prepare per-bin writers (lazy-open)
        bin_writers = {}
        tmpdir_path = Path(tmpdir)
        tmpdir_path.mkdir(parents=True, exist_ok=True)

        # Create an iterator from the reader and skip to start_idx
        it = iter(reader)
        for _ in range(start_idx):
            next(it, None)

        # If a barcode DB path was provided, open it read-only for lookups
        conn = None
        cur = None
        lmdb_env = None
        lmdb_txn = None
        if barcode_db_path is not None:
            if barcode_db_path.endswith('.lmdb'):
                lmdb_env, lmdb_txn = open_lmdb_readonly(barcode_db_path)
            else:
                conn, cur = open_sqlite_readonly(barcode_db_path)

        # Process assigned chunk using the iterator
        for read_idx, (read1, read2) in enumerate(it, start_idx):
            if read_idx >= end_idx:
                break

            name_and_pos = read1.name.split(maxsplit=1)[0]
            uncorrected_barcode_seq = uncorrected_barcode_reader.get_barcode(name_and_pos)
            if lmdb_txn is not None:
                corrected_barcode_seq = lookup_lmdb(lmdb_txn, uncorrected_barcode_seq)
            elif cur is not None:
                corrected_barcode_seq = lookup_canonical(cur, uncorrected_barcode_seq)
            else:
                corrected_barcode_seq = None

            raw_barcode_id = f"{sequence_tag}:Z:{uncorrected_barcode_seq}"
            corr_barcode_id = f"{barcode_tag}:Z:{corrected_barcode_seq}"

            read1.name = f"{name_and_pos}_{raw_barcode_id}_{corr_barcode_id}"
            read2.name = read1.name


            # If mapper is ema and output_bins requested, write into deterministic per-bin files
            if mapper == "ema" and output_bins is not None and corrected_barcode_seq is not None:
                # Compute deterministic bin for this barcode
                bin_idx = barcode_to_bin(corrected_barcode_seq, nr_bins)
                # Lazy open writer for this bin
                if bin_idx not in bin_writers:
                    bin_file = tmpdir_path / f"bin_{str(bin_idx).zfill(3)}_rank_{rank}.fastq"
                    bw = dnaio.open(str(bin_file), interleaved=True, mode="w", fileformat="fastq")
                    bin_writers[bin_idx] = bw

                # Write interleaved FASTQ to the bin writer
                bin_writers[bin_idx].write(read1, read2)
                summary["Read pairs written"] += 1
                continue

            # Fallback: write using output handler if available (non-EMA or no binning)
            if corrected_barcode_seq is None:
                summary["Reads missing barcode"] += 1
                # If non-barcoded outputs are desired, use Output.write_nobc
                if output_handler is not None:
                    output_handler.write_nobc(read1, read2)
                continue

            # # For non-EMA mappers, update headers similarly to original behavior and write
            # raw_barcode_id = f"{sequence_tag}:Z:{uncorrected_barcode_seq}"
            # corr_barcode_id = f"{barcode_tag}:Z:{corrected_barcode_seq}"
            # # Update read headers
            # read1.name = f"{name_and_pos}_{raw_barcode_id}_{corr_barcode_id}"
            # read2.name = read1.name
            
            if output_handler is not None:
                output_handler.write(read1, read2)
                summary["Read pairs written"] += 1
            else:
                # Write to a simple tmp file when no output handler
                tmpfile = tmpdir_path / f"rank_{rank}_output.fastq"
                with dnaio.open(str(tmpfile), interleaved=True, mode="a", fileformat="fastq") as w:
                    w.write(read1, read2)
                summary["Read pairs written"] += 1

        # Close any open bin writers
        for bw in bin_writers.values():
            bw.close()
        # Close DB connection if opened
        if conn is not None:
            conn.close()
        if lmdb_env is not None:
            lmdb_env.close()

def merge_rank_outputs(tmpdir, final_output, num_ranks, nr_bins=None):
    """Merge temporary outputs from all ranks into final output files.

    If nr_bins is provided, assemble per-bin files named according to
    Output.BIN_FASTQ_TEMPLATE into the directory `final_output`.
    Otherwise, concatenate per-rank tmp files into `final_output`.
    """
    if nr_bins:
        final_dir = Path(final_output)
        final_dir.mkdir(parents=True, exist_ok=True)
        for bin_idx in range(nr_bins):
            bin_str = str(bin_idx).zfill(3)
            final_name = final_dir / Output.BIN_FASTQ_TEMPLATE.replace("*", bin_str)
            with open(final_name, 'wb') as outfile:
                for rank_num in range(num_ranks):
                    tmp_file = Path(tmpdir) / f"bin_{bin_str}_rank_{rank_num}.fastq"
                    if tmp_file.exists():
                        # binary mode to preserve exact bytes
                        with open(tmp_file, 'rb') as infile:
                            outfile.write(infile.read())
                        tmp_file.unlink()
    else:
        # single output file merge
        with open(final_output, 'w') as outfile:
            for rank_num in range(num_ranks):
                tmp_file = f"{tmpdir}/rank_{rank_num}_output.fastq"
                if os.path.exists(tmp_file):
                    with open(tmp_file) as infile:
                        outfile.write(infile.read())
                    os.remove(tmp_file)

def setup_output(*args, **kwargs):
    """Wrapper around existing Output class initialization"""
    return Output(*args, **kwargs)


class BarcodeReader:
    def __init__(self, filename):
        self._cache = {}
        self._file = dnaio.open(filename, mode="r")
        self.barcodes = iter(self._file)

    def get_barcode(self, read_name, maxiter=128):
        if read_name in self._cache:
            return self._cache.pop(read_name)

        for barcode in islice(self.barcodes, maxiter):
            barcode_id = barcode.name.partition(" ")[0]
            # If read_name in next pair then parser lines are synced --> drop cache.
            if read_name == barcode_id:
                self._cache.clear()
                return barcode.sequence

            self._cache[barcode_id] = barcode.sequence

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._file.close()

class Output:
    """
    Output handler for different output formats required by different mappers.
    """
    BIN_FASTQ_TEMPLATE = "ema-bin-*"  # Same name as in `ema preproc`.

    def __init__(self, file1=None, file2=None, interleaved=False, file_nobc1=None, file_nobc2=None, mapper=None,
                 bins_dir=None, nr_bins=None):
        self._mapper = mapper

        self._bin_nr = 0
        self._reads_written = 0
        self._bin_size = None
        self._bins_dir = bins_dir
        self._nr_bins = nr_bins
        self._barcode_bin_map = {}
        self._open_bins = None
        self._prev_heap = None
        self._bin_filled = True
        self._pre_write = lambda *args: None
        self._post_write = lambda *args: None

        self._open_file = None
        if file1 is not None:
            self._open_file = self._setup_single_output(file1, file2, interleaved)
        elif bins_dir is not None:
            self._pre_write = self._open_new_bin_if_full
            self._post_write = self._check_bin_full
        else:
            sys.exit("Either file1 or bins_dir need to be provided.")

        self._open_file_nobc = None
        if file_nobc1 is not None:
            self._open_file_nobc = self._setup_single_output(file_nobc1, file_nobc2,
                                                             interleaved=file_nobc2 is None)

        # Setup writers based on mapper.
        if self._mapper == "lariat":
            self._write = self._write_lariat
            self._write_nobc = self._write_nobc_lariat
        else:
            self._write = self._write_default
            self._write_nobc = self._write_nobc_default

    def set_bin_size(self, value):
        self._bin_size = value

    def _setup_single_output(self, file1, file2, interleaved):
        if self._mapper == "lariat":
            if file2 is not None:
                Path(file2).touch()
            return xopen(file1, mode='w')
        else:
            return dnaio.open(file1, file2=file2, interleaved=interleaved, mode="w", fileformat="fastq")

    def _get_bin_name(self):
        bin_nr_str = str(self._bin_nr).zfill(3)
        file_name = self._bins_dir / Output.BIN_FASTQ_TEMPLATE.replace("*", bin_nr_str)
        self._bin_nr += 1
        return file_name

    def _open_new_bin(self):
        if self._open_file is not None:
            self._open_file.close()

        file_name = self._get_bin_name()
        self._open_file = dnaio.open(file_name, interleaved=True, mode="w", fileformat="fastq")

    def _open_new_bin_if_full(self, heap):
        # Start a new bin if the current is full while not splitting heaps over separate bins
        if self._bin_filled and heap != self._prev_heap:
            self._bin_filled = False
            self._open_new_bin()
            logger.debug(f"Bin overflow = {self._reads_written} ")

        self._qprev_heap = heap

    def _open_next_bin(self):
        if self._open_bins is None:
            # Open all bins
            self._open_bins = []
            for i in range(self._nr_bins):
                file_name = self._get_bin_name()
                self._open_bins.append(xopen(file_name, "w"))

            self._open_bins = cycle(self._open_bins)

        self._open_file = next(self._open_bins)

    def _check_bin_full(self):
        if self._reads_written > self._bin_size:
            self._bin_filled = True
            self._reads_written = 0

    def _write_default(self, read1, read2):
        self._open_file.write(read1, read2)

    def _write_lariat(self, read1, _):
        self._open_file.write(read1)

    def _write_ema_special(self, read1, read2, barcode):
        line = f"{barcode} @{read1.name} {read1.sequence} {read1.qualities} {read2.sequence} {read2.qualities}\n"
        self._open_file.write(line)

    def write(self, read1, read2=None, heap=None):
        self._pre_write(heap)

        # Write reads to output file
        self._write(read1, read2)
        self._reads_written += 1

        self._post_write()

    def write_ema_special(self, read1, read2, barcode):
        assert self._nr_bins is not None, "Please set nr_bins to use ema special format"
        self._open_file = self._barcode_bin_map.get(barcode)
        if self._open_file is None:
            self._open_next_bin()
            self._barcode_bin_map[barcode] = self._open_file

        self._write_ema_special(read1, read2, barcode)

    def _write_nobc_default(self, read1, read2):
        self._open_file_nobc.write(read1, read2)

    def _write_nobc_lariat(self, read1, _):
        self._open_file_nobc.write(read1)

    def write_nobc(self, read1, read2=None):
        if self._open_file_nobc is not None:
            self._write_nobc(read1, read2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._open_file is not None:
            self._open_file.close()
        if self._open_file_nobc is not None:
            self._open_file_nobc.close()

        if self._open_bins is not None:
            for file in self._open_bins:
                if file.closed:
                    break
                else:
                    file.close()

class ChunkHandler:
    def __init__(self, tmpdir, chunk_size: int = 100_000):
        # Required for ema sorted output
        # Inspired by: https://stackoverflow.com/questions/56948292/python-sort-a-large-list-that-doesnt-fit-in-memory
        self._output_chunk = []
        self._chunk_size = chunk_size
        self._chunk_id = 0
        self._tmpdir = Path(tempfile.mkdtemp(prefix="tagfastq_sort", dir=tmpdir))
        self._chunk_file_template = "chunk_*.tsv"
        self._chunk_files = []
        self._chunk_sep = "\t"
        self._tmp_writer = self.create_writer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tmp_writer.close()

    def create_writer(self):
        if hasattr(self, "_tmp_writer"):
            self._tmp_writer.close()
        tmpfile = self._tmpdir / self._chunk_file_template.replace("*", str(self._chunk_id))
        self._chunk_files.append(tmpfile)
        self._chunk_id += 1
        return open(tmpfile, "w")

    def build_chunk(self, line: str):
        """Add entry to write to temporary file, first argument should be the heap index"""
        self._output_chunk.append(line)

        if len(self._output_chunk) > self._chunk_size:
            self.write_chunk()
            self._tmp_writer = self.create_writer()

    def write_chunk(self):
        self._output_chunk.sort(key=self._get_heap)
        self._tmp_writer.writelines(self._output_chunk)
        self._output_chunk *= 0  # Clear list faster than list.clear(). See https://stackoverflow.com/a/44349418

    def parse_chunks(self):
        if not self._tmp_writer.closed:
            self._tmp_writer.close()

        with ExitStack() as chunkstack:
            logger.info("Opening chunks for merge")
            chunks = [chunkstack.enter_context(chunk.open(mode="r")) for chunk in self._chunk_files]
            logger.info("Merging chunks")
            for entry in merge(*chunks, key=self._get_heap):
                yield entry.strip().split(self._chunk_sep)

    def _get_heap(self, x):
        return int(x.split(self._chunk_sep)[0])

def add_arguments(parser):
    parser.add_argument(
        "uncorrected_barcodes",
        help="FASTQ/FASTA for uncorrected barcodes."
    )
    parser.add_argument(
        "corrected_barcodes",
        help="FASTQ/FASTA for error corrected barcodes. Currently accepts output from starcode "
             "clustering with '--print-clusters' enabled."
    )
    parser.add_argument(
        "input1",
        help="Input FASTQ/FASTA file. Assumes to contain read1 if given with second input file. "
             "If only input1 is given, input is assumed to be an interleaved. If reading from stdin"
             "is requested use '-' as a placeholder."
    )
    parser.add_argument(
        "input2", nargs='?',
        help="Input  FASTQ/FASTA for read2 for paired-end read. Leave empty if using interleaved."
    )
    output = parser.add_mutually_exclusive_group(required=False)
    output.add_argument(
        "--output1", "--o1",
        help="Output FASTQ/FASTA file name for read1. If not specified the result is written to "
             "stdout as interleaved. If output1 given but not output2, output will be written as "
             "interleaved to output1."
    )
    parser.add_argument(
        "--output2", "--o2",
        help="Output FASTQ/FASTA name for read2. If not specified but --o1/--output1 given the "
             "result is written as interleaved."
    )
    parser.add_argument(
        "--output-nobc1", "--n1",
        help="Only for ema! Output FASTQ/FASTA file name to write non-barcoded read1 reads. "
             "If output_nobc1 given but not output_nobc2, output will be written as interleaved to "
             "output_nobc1."
    )
    parser.add_argument(
        "--output-nobc2", "--n2",
        help="Only for ema! Output FASTQ/FASTA file name to write non-barcoded read2 reads."
    )
    output.add_argument(
        "--output-bins",
        help=f"Output interleaved FASTQ split into bins named '{Output.BIN_FASTQ_TEMPLATE}' in the provided "
             f"directory. Entries will be grouped based on barcode. Only used for ema mapping."
    )
    parser.add_argument(
        "--nr-bins", type=int, default=100,
        help="Number of bins to split reads into when using the '--output-bins' alternative. Default: %(default)s."
    )
    parser.add_argument(
        "--tmpdir", type=str, default="./",
        help="tmpdir"
    )
    parser.add_argument(
        "-b", "--barcode-tag", default="BX",
        help="SAM tag for storing the error corrected barcode. Default: %(default)s."
    )
    parser.add_argument(
        "-s", "--sequence-tag", default="RX",
        help="SAM tag for storing the uncorrected barcode sequence. Default: %(default)s."
    )
    parser.add_argument(
        "-m", "--mapper", default="bowtie2", choices=ACCEPTED_READ_MAPPERS,
        help="Specify read mapper for labeling reads with barcodes. Selecting 'ema' or 'lariat' produces output "
             "required for these particular mappers. Default: %(default)s."
    )
    parser.add_argument(
        "-c", "--min-count", default=0, type=int,
        help="Minimum number of reads per barcode to tag read name. Default: %(default)s."
    )
    parser.add_argument(
        "-p", "--pattern-match",
        help="IUPAC barcode string to match against corrected barcodes e.g. for DBS it is usualy "
             "BDHVBDHVBDHVBDHVBDHV. Non-matched barcodes will be removed."
    )
    parser.add_argument(
        "--sample-nr", type=int, default=1,
        help="Sample number to append to barcode string. Default: %(default)s."
    )
    parser.add_argument(
        "--mpi-chunk-size",
        type=int,
        default=1000000,
        help="Number of reads to process in each MPI chunk. Default: %(default)s"
    )
    parser.add_argument(
        "--build-db",
        action="store_true",
        help="Only build the barcode LMDB in --tmpdir and exit (rank 0 does the build)."
    )
    parser.add_argument(
        "--lmdb-map-size",
        type=int,
        default=None,
        help="Optional LMDB map size in bytes to use when building the LMDB (default: 1<<34)."
    )
