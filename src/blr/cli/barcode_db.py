from pathlib import Path
import pandas as pd
import sqlite3
import lmdb

# Local replacements of helper functions to avoid circular imports

def match_template(sequence: str, template) -> bool:
    if len(sequence) != len(template):
        return False
    for base, accepted_bases in zip(sequence, template):
        if base not in accepted_bases:
            return False
    return True


def scramble(seqs, maxiter=10):
    swapped = True
    iteration = 1
    start_from = 0
    while swapped and iteration < maxiter:
        iteration += 1
        swapped = False
        swap_pos = []
        for i in range(start_from, len(seqs) - 2):
            if seqs[i][:16] == seqs[i + 1][:16]:
                swap_pos.append(i)
                seqs[i + 2], seqs[i + 1] = seqs[i + 1], seqs[i + 2]
                swapped = True
        start_from = min(swap_pos) if swap_pos else 0

    return


def build_barcode_sqlite(clusters_file: str, db_path: str, summary, mapper, template=None, min_count=0,
                         chunksize: int = 10000):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("CREATE TABLE IF NOT EXISTS mapping(raw_seq TEXT PRIMARY KEY, canonical_seq TEXT);")
    insert_sql = "INSERT OR REPLACE INTO mapping(raw_seq, canonical_seq) VALUES (?, ?);"

    total_clusters = 0
    total_reads = 0
    kept_clusters = 0
    kept_reads = 0
    canonical_seqs = []

    cols = ["canonical_seq", "size", "cluster_seqs"]
    for chunk in pd.read_csv(clusters_file, sep="\t", names=cols, dtype={"canonical_seq": str, "size": int, "cluster_seqs": str}, chunksize=chunksize):
        total_clusters += len(chunk)
        total_reads += chunk["size"].sum()

        if template:
            mask = (chunk["size"] >= min_count) & (chunk["canonical_seq"].apply(match_template, template=template))
        else:
            mask = (chunk["size"] >= min_count)

        filtered = chunk[mask]
        kept_clusters += len(filtered)
        kept_reads += filtered["size"].sum()

        rows_to_insert = []
        for canonical, seqs in zip(filtered["canonical_seq"], filtered["cluster_seqs"]):
            canonical_seqs.append(canonical)
            if not isinstance(seqs, str):
                continue
            for raw in seqs.split(","):
                rows_to_insert.append((raw, canonical))

        if rows_to_insert:
            cur.executemany(insert_sql, rows_to_insert)
            conn.commit()

    summary["Corrected barcodes"] = total_clusters
    summary["Reads with corrected barcodes"] = int(total_reads)
    summary["Barcodes not passing filters"] = total_clusters - kept_clusters
    summary["Reads with barcodes not passing filters"] = int(total_reads - kept_reads)

    heap_index = {}
    if mapper in ["ema", "lariat"]:
        canonical_list = canonical_seqs
        if mapper == "ema":
            scramble(canonical_list, maxiter=100)
        heap_index = {seq: nr for nr, seq in enumerate(canonical_list)}

    conn.close()
    return db_path, heap_index


def build_barcode_lmdb(clusters_file: str, lmdb_path: str, summary, mapper, template=None, min_count=0,
                       chunksize: int = 10000, map_size: int = 1 << 34):
    lmdb_dir = Path(lmdb_path)
    if lmdb_dir.exists():
        for child in lmdb_dir.iterdir():
            try:
                child.unlink()
            except Exception:
                pass
        try:
            lmdb_dir.rmdir()
        except Exception:
            pass
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(lmdb_dir), map_size=map_size)

    total_clusters = 0
    total_reads = 0
    kept_clusters = 0
    kept_reads = 0
    canonical_seqs = []

    cols = ["canonical_seq", "size", "cluster_seqs"]
    # Use a write transaction per chunk
    with env.begin(write=True) as txn:
        for chunk in pd.read_csv(clusters_file, sep="\t", names=cols, dtype={"canonical_seq": str, "size": int, "cluster_seqs": str}, chunksize=chunksize):
            total_clusters += len(chunk)
            total_reads += chunk["size"].sum()

            if template:
                mask = (chunk["size"] >= min_count) & (chunk["canonical_seq"].apply(match_template, template=template))
            else:
                mask = (chunk["size"] >= min_count)

            filtered = chunk[mask]
            kept_clusters += len(filtered)
            kept_reads += filtered["size"].sum()

            for canonical, seqs in zip(filtered["canonical_seq"], filtered["cluster_seqs"]):
                canonical_seqs.append(canonical)
                if not isinstance(seqs, str):
                    continue
                for raw in seqs.split(","):
                    txn.put(raw.encode("ascii"), canonical.encode("ascii"))

            txn.commit()
            txn = env.begin(write=True)

    summary["Corrected barcodes"] = total_clusters
    summary["Reads with corrected barcodes"] = int(total_reads)
    summary["Barcodes not passing filters"] = total_clusters - kept_clusters
    summary["Reads with barcodes not passing filters"] = int(total_reads - kept_reads)

    heap_index = {}
    if mapper in ["ema", "lariat"]:
        canonical_list = canonical_seqs
        if mapper == "ema":
            scramble(canonical_list, maxiter=100)
        heap_index = {seq: nr for nr, seq in enumerate(canonical_list)}

    env.sync()
    env.close()
    return str(lmdb_dir), heap_index


def open_sqlite_readonly(db_path: str):
    uri = f'file:{db_path}?mode=ro'
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    cur = conn.cursor()
    return conn, cur


def open_lmdb_readonly(lmdb_path: str):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, max_readers=256)
    txn = env.begin(buffers=False)
    return env, txn


def lookup_canonical(cur, raw_seq: str):
    cur.execute("SELECT canonical_seq FROM mapping WHERE raw_seq = ?;", (raw_seq,))
    r = cur.fetchone()
    return r[0] if r else None


def lookup_lmdb(txn, raw_seq: str):
    val = txn.get(raw_seq.encode("ascii"))
    return val.decode("ascii") if val is not None else None
