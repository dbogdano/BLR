"""Small standalone tag-and-bin utilities.

This package provides two lightweight scripts:
- `bin_builder`: greedy largest-first packing that writes a canonical->bin TSV
- `tag_and_bin`: tags FASTQ reads and writes them directly into per-bin FASTQ files

Dependencies (recommended): pandas, dnaio, xopen
"""

__all__ = ["bin_builder", "tag_and_bin"]
