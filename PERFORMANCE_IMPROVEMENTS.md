# Performance Improvements for tagfastq with --bin-map

## Problem
Running `tagfastq` with the `--bin-map` option on an 82GB gzipped paired-end FASTQ file was taking >6 hours with 100GB of RAM.

## Root Causes Identified

### 1. **No I/O Buffering** (CRITICAL)
When writing to 100+ bin files simultaneously, Python's default unbuffered text I/O was causing excessive system calls. Each `write()` operation triggered immediate disk I/O.

### 2. **Inefficient String Formatting**
Using f-strings to build tab-separated lines for billions of reads created unnecessary overhead:
```python
# Before (slow)
line = f"{canonical_barcode}\t{heap_idx}\t{read1.name}\t{read1.sequence}\t{read1.qualities}\t{read2.sequence}\t{read2.qualities}\n"
```

### 3. **Lack of Write Batching**
Every read was written immediately to disk rather than batching writes in memory.

## Optimizations Implemented

### 1. Increased File Buffer Size
Added 1MB buffer to file handles when opening bin chunk files:
```python
fh = open(tmp_name, "w", buffering=1024*1024)
```

### 2. In-Memory Write Buffering
Implemented per-bin in-memory buffers that accumulate 10,000 reads before flushing:
```python
self._bin_buffers = []  # One buffer list per bin
self._buffer_size = 10000  # Tunable parameter
```

### 3. Optimized String Building
Replaced f-strings with `str.join()` for better performance:
```python
# After (faster)
line = "\t".join([canonical_barcode, str(heap_idx), read1.name, 
                 read1.sequence, read1.qualities, read2.sequence, read2.qualities]) + "\n"
```

### 4. Batch Writes with `writelines()`
Buffer contents are flushed using `writelines()` which is more efficient than multiple `write()` calls:
```python
if len(buffer) >= self._buffer_size:
    fh.writelines(buffer)
    buffer.clear()
```

### 5. Proper Buffer Flushing on Exit
Ensures all buffered data is written before closing files in the `__exit__` method.

## Expected Performance Impact

### Time Reduction
- **I/O buffering**: 2-3x speedup (reducing system calls by ~99%)
- **Write batching**: 1.5-2x additional speedup (reducing Python overhead)
- **String optimization**: ~10-15% improvement
- **Combined**: Expected 3-6x total speedup, reducing runtime from >6 hours to **1-2 hours**

### Memory Impact
- Additional memory: ~10MB per 100 bins (10k reads × 100 bins × ~1KB/read)
- Minimal compared to 100GB available

## Tuning Parameters

If you need to adjust the performance/memory tradeoff:

```bash
# The buffer size can be tuned by modifying _buffer_size in _open_all_bins()
# Current: 10,000 reads per bin
# For lower memory: 5,000 reads
# For faster processing (if you have memory): 50,000 reads
```

## Additional Recommendations

### 1. Use LMDB Barcode Database
For very large barcode sets, use `--barcode-db` with LMDB to avoid loading the full raw→canonical mapping into RAM:

```bash
# First, build the LMDB database
blr tagfastq --build-db --lmdb-map-size $((1<<35)) \
    uncorrected_barcodes.fq corrected_barcodes.tsv

# Then use it during tagging
blr tagfastq --barcode-db barcode_mapping.lmdb --bin-map bin_mapping.tsv \
    uncorrected_barcodes.fq corrected_barcodes.tsv input1.fq input2.fq \
    --output-bins output_bins/ --nr-bins 100 --mapper ema
```

### 2. Adjust Chunk Size
The `--chunk-size` parameter controls memory usage for EMA/lariat sorting. Lower values reduce peak memory:

```bash
--chunk-size 50000  # Lower memory footprint
--chunk-size 200000 # Default, balanced
```

### 3. Use Faster Storage
- **Local NVMe SSD** is best for bin output
- Avoid network filesystems (NFS, CIFS) for the `--output-bins` directory
- If using network storage, increase `_buffer_size` to 50,000-100,000

### 4. Monitor Progress
The code uses `tqdm` for progress tracking. Watch the "Read pairs processed" counter to estimate completion time.

## Testing

To verify the improvements work correctly:

```bash
# Run with a small test dataset first
blr tagfastq --bin-map test_bin_map.tsv \
    test_uncorrected.fq test_corrected.tsv \
    test_input1.fq test_input2.fq \
    --output-bins test_output/ --nr-bins 10 --mapper ema
```

## Monitoring Performance

Watch for these indicators:
- **CPU usage**: Should be mostly I/O-wait, not Python compute
- **Disk I/O**: Should show burst writes every ~10k reads (when buffers flush)
- **Memory**: Should remain stable, not growing continuously

If memory grows continuously, there may be a buffer leak - check that all buffers are properly flushed.
