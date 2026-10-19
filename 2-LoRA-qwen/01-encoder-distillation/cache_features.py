#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from sharm_image21.core import (ARTIFACTS, CACHE, FeatureCache, FrozenEncoder,
                                atomic_json, digest, model_identity, read_jsonl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["teacher", "student"])
    parser.add_argument("--corpus", type=Path, default=ARTIFACTS / "corpus.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-root", type=Path, default=CACHE)
    args = parser.parse_args()
    lock_dir = args.cache_root / args.kind
    lock_dir.mkdir(parents=True, exist_ok=True)
    writer_lock = (lock_dir / ".writer.lock").open("a")
    fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX)
    corpus = read_jsonl(args.corpus)
    rows = corpus[:args.limit] if args.limit else corpus
    identity = {"encoder": model_identity(args.kind), "corpus": digest(corpus)}
    cache = FeatureCache(lock_dir, identity)
    pending = [row for row in rows if row["id"] not in cache]
    if not pending:
        print(f"Cache {args.kind} complet pour {len(rows)} exemples.", flush=True)
        return
    encoder = FrozenEncoder(args.kind, args.gpu_memory)
    pending.sort(key=lambda row: row.get("tokens_unwrapped", len(row["text"])))
    started, buffered, completed = time.monotonic(), [], len(rows)-len(pending)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start+args.batch_size]
        output = encoder.encode([row["text"] for row in batch])
        buffered.extend((row["id"], values, token_ids) for row, (values, token_ids) in zip(batch, output))
        completed += len(batch)
        if len(buffered) >= 32 or start+args.batch_size >= len(pending):
            cache.write(buffered)
            buffered = []
            elapsed = time.monotonic()-started
            report = {"stage": f"cache_{args.kind}", "completed": completed, "total": len(rows),
                      "seconds": elapsed, "updated_at": time.time(),
                      "examples_per_second": (start+len(batch))/max(elapsed, 1e-6)}
            status_root = ARTIFACTS if args.cache_root.resolve() == CACHE.resolve() else args.cache_root
            atomic_json(status_root / f"cache-{args.kind}-status.json", report)
            print(f"{args.kind}: {completed}/{len(rows)} — {elapsed:.1f}s", flush=True)
    cache.require(rows)


if __name__ == "__main__":
    main()
