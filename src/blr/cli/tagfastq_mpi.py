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
):
    summary = Summary()
    
    # Only rank 0 reads and broadcasts the corrected barcodes
    if rank == 0:
        logger.info("Map clusters")
        template = [set(IUPAC[base]) for base in pattern_match] if pattern_match else []
        seq_to_barcode, heap = map_corrected_barcodes(corrected_barcodes, summary, mapper, template, min_count)
    else:
        seq_to_barcode = None
        heap = None
    
    # Broadcast the corrected barcodes dictionary and heap to all processes
    seq_to_barcode = comm.bcast(seq_to_barcode, root=0)
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
        input1, input2, start_idx, end_idx, seq_to_barcode, heap,
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

def process_reads_chunk(input1, input2, start_idx, end_idx, seq_to_barcode, heap,
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

        # Skip to start_idx
        for _ in range(start_idx):
            next(reader)

        # Process assigned chunk
        for read_idx, (read1, read2) in enumerate(reader, start_idx):
            if read_idx >= end_idx:
                break

            name_and_pos = read1.name.split(maxsplit=1)[0]
            uncorrected_barcode_seq = uncorrected_barcode_reader.get_barcode(name_and_pos)
            corrected_barcode_seq = seq_to_barcode.get(uncorrected_barcode_seq, None)

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

            # For non-EMA mappers, update headers similarly to original behavior and write
            raw_barcode_id = f"{sequence_tag}:Z:{uncorrected_barcode_seq}"
            corr_barcode_id = f"{barcode_tag}:Z:{corrected_barcode_seq}"
            # Update read headers
            read1.name = f"{name_and_pos}_{raw_barcode_id}_{corr_barcode_id}"
            read2.name = read1.name
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

# Include all existing classes (BarcodeReader, Output, ChunkHandler) and helper functions 
# from the original tagfastq.py...

def add_arguments(parser):
    # Add all existing arguments from original tagfastq.py
    parser.add_argument(
        "--mpi-chunk-size",
        type=int,
        default=1000000,
        help="Number of reads to process in each MPI chunk. Default: %(default)s"
    )
    # Include all other existing arguments...