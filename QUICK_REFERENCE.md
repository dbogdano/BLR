# Quick Reference: tagfastq Performance Optimizations

## What Changed?
✓ Added **I/O buffering** (1MB file buffers)
✓ Implemented **write batching** (10k reads per bin)
✓ Optimized **string operations** (str.join vs f-strings)
✓ Proper **buffer flushing** on exit

## Expected Results
- **Before**: >6 hours for 82GB input
- **After**: 1-2 hours (3-6x faster)
- **Memory**: +10MB overhead (negligible)

## No Interface Changes
Use your existing commands - the optimization is automatic:

```bash
blr tagfastq --bin-map bin_mapping.tsv \
    uncorrected_barcodes.fq corrected_barcodes.tsv \
    input1.fq input2.fq \
    --output-bins output_bins/ --nr-bins 100 --mapper ema
```

## Troubleshooting

### If still slow (>2 hours):
1. Check disk I/O: Use SSD, not network storage
2. Increase buffer size in code: Change `_buffer_size = 10000` to `50000`
3. Use LMDB: `--barcode-db barcode_mapping.lmdb`

### If memory issues:
1. Decrease buffer size: Change `_buffer_size = 10000` to `5000`
2. Use `--chunk-size 50000` flag

### Monitor progress:
```bash
# The tqdm progress bar shows "Read pairs processed"
# Watch for steady progress without stalls
```

## Key Files
- Modified: [src/blr/cli/tagfastq.py](src/blr/cli/tagfastq.py#L618-L766)
- Docs: [PERFORMANCE_FIX_SUMMARY.md](PERFORMANCE_FIX_SUMMARY.md)

## Technical Details
See [PERFORMANCE_IMPROVEMENTS.md](PERFORMANCE_IMPROVEMENTS.md) for complete analysis.
