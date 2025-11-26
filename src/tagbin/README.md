TagBin — compact greedy binning + tag-and-bin utilities

This small package provides:

- `bin_builder.py`: greedy largest-first bin packing to generate a canonical->bin TSV.
- `tag_and_bin.py`: tag reads using raw barcodes and write directly into per-bin FASTQ files.

Recommended dependencies: `pandas`, `dnaio`, `xopen`.

Usage examples:

Build mapping:
```bash
PYTHONPATH=src python3 -m tagbin.bin_builder clusters.tsv --nr-bins 1000 --out-mapping barcode_bins.tsv
```

Tag and bin reads:
```bash
PYTHONPATH=src python3 -m tagbin.tag_and_bin raw_barcodes.fastq clusters.tsv reads_R1.fastq reads_R2.fastq --out-dir outbins --bin-map barcode_bins.tsv
```
