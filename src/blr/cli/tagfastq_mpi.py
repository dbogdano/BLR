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
"""
MPI support has been removed from this project.

The original `tagfastq_mpi.py` module implemented an MPI-distributed version
of the `tagfastq` tool. This repository no longer provides MPI functionality.

To avoid import-time errors on systems without MPI or mpi4py installed,
we provide a small shim that informs users the MPI variant has been removed.
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser


def add_arguments(parser: ArgumentParser):
    """Present a compatible CLI signature for discovery; actual execution is disabled."""
    parser.add_argument("--help-mpi", action="store_true", help="MPI support removed; see documentation.")


def main(args):
    sys.stderr.write("Error: MPI support has been removed from BLR.\n")
    sys.stderr.write("Use the serial `tagfastq` command or the deterministic bin-mapper instead.\n")
    return 1
    start_idx = rank * chunk_size
