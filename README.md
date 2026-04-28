# VEC 2.0

A dead simple vector database that lives on your GPU. Now with a sidecar payload field — every record is `(vector, label, ≤100KB blob)`. The MySQL sidekick is gone; one exe replaces both.

Store millions of embeddings in VRAM. Find the nearest ones in milliseconds. Single exe, no dependencies, no configuration.

```bash
# Start a database
vec mydb 1024

# That's it. It's listening on port 1920.
```

No GPU? No problem. `vec-cpu` does the same thing using RAM.

```bash
vec-cpu mydb 1024
```

> **Upgrading from 1.x?** See `MIGRATION.md`. Server protocol is a clean break — old SDKs won't work. `.tensors` and `.meta` files load unchanged; `.data` is a new sidecar that appears on first save once you store payloads.

---

## Protocol

Binary only. Every request and every response is a length-prefixed binary frame over TCP, named pipe, or Unix socket.

```
request:  F0 <2B ns_len> [ns] <CMD> <2B label_len> [label] <4B body_len> [body]
response: <1B status> <4B body_len> [body]        ; status 0=ok, 1=err
```

15 commands, one envelope. See `PROTOCOL-2.0.md` for the byte-exact spec or `sdk/README.md` for the quick reference.

Use the SDK libraries — there is no text interface.

---

## What it does

- **PUSH** — store a vector, optionally with a label and a ≤100KB data payload (data requires label). Returns the slot index.
- **QUERY** — nearest-neighbor search; metric byte selects L2 or cosine.
- **QID** — like QUERY but the query is an existing stored vector (by index or label).
- **GET** — retrieve records by index, by label (may yield multiple), or batch by index list.
- **SET_DATA / GET_DATA** — manage the per-slot payload independently of the vector.
- **UPDATE** — overwrite a vector in place (label and data untouched).
- **LABEL** — set or clear a slot's label.
- **DELETE** — tombstone a slot (also frees its label and data).
- **UNDO** — remove the last PUSH (also frees its label and data).
- **CLUSTER** — DBSCAN over the full set.
- **DISTINCT** — farthest-point sampling (k most spread-out vectors).
- **REPRESENT** — one most-distinct member per cluster.
- **INFO** — database metadata (dim, count, format, CRC, name, protocol version).
- **SAVE** — flush to disk; returns saved count + CRC.

## Result shape

QUERY/QID/GET take a 1-byte **shape mask** that controls what each result record carries:

- `0x01` vector
- `0x02` label
- `0x04` data
- `0x07` all three (default)

Skip what you don't need to keep responses lean.

## L2 vs Cosine

- **L2** (squared Euclidean) — "how far apart?" Use for vision models (DINOv2, ArcFace).
- **Cosine** — "looking the same direction?" Use for text models (BGE, MiniLM, CLIP).

QUERY and QID take a metric byte (0=L2 default, 1=cosine). CLUSTER/DISTINCT/REPRESENT take the same byte.

---

## Multiple databases

Run them behind a router:

```bash
# Start databases without TCP (pipe/socket only)
vec --notcp tools 1024
vec --notcp conversations 1024

# Route them all through one port
vec --route 1920
```

Or let deploy mode do it all:

```bash
# Auto-discover all .tensors files
vec deploy

# Custom port
vec deploy 1920

# Explicit schema
vec --deploy=tools:1024,conversations:1024,faces:512:f16 1920
```

The SDK sets a namespace on the client. The router strips it from the frame and forwards to the correct backend.

---

## Housekeeping

```bash
# Delete an entire database
vec face --delete

# Check file integrity (dry run)
vec mydb --check

# Repair a corrupt database
vec mydb --repair

# Auto-detect .tensors in current directory
vec

# Load a specific file
vec mydb.tensors

# fp16 mode — half the VRAM, double the capacity (GPU only)
vec mydb 1024:f16
```

---

## What it looks like

```
NVIDIA GeForce RTX 3060 (12.0 GB)
===================================================================
  database      mydb
  format        f32, 1024 dim
  records       4.2m total, 4.1m active, 100.0k deleted
  file size     16.1 GB
  modified      12 minutes ago (14:20)
  capacity      2.9m max, 1.7m remaining (58.6%)
  checksum      NOMITOPO (0xA3F291B7) ok
===================================================================
```

---

## Performance

1024-dim vectors:

```
             GPU (RTX 3060)    CPU
  10K        ~0.2 ms           ~2 ms
  100K       ~1.5 ms           ~20 ms
  1M         ~14 ms            ~200 ms
```

## Capacity

fp32, 1024 dimensions:

```
  8 GB VRAM     1.9M vectors
  12 GB VRAM    2.9M vectors
  24 GB VRAM    5.8M vectors
```

Lower dimensions or fp16 = more capacity.

---

## Supported GPUs

RTX 2000 series and newer (Turing, Ampere, Ada Lovelace). RTX 2060 through 4090, plus T4, A100, L40.

GTX 1000 series and older won't work. AMD/Intel GPUs won't work. Use `vec-cpu` instead.

---

## Build

```bash
# Windows
build.bat

# Linux
./build.sh
```

Requires NVIDIA CUDA Toolkit 12.x for `vec`. `vec-cpu` just needs a C++ compiler.

---

## Good to know

- **Brute force.** Every query scans every vector. Exact results, zero approximation.
- **GPU top-K.** Above 100K vectors, a CUDA kernel finds the top results on GPU.
- **All RAM.** Vectors in VRAM (or RAM for vec-cpu), labels and data alongside them. No mmap, no lazy paging — disk is touched only on startup load and explicit SAVE.
- **Indices are permanent.** Slot 42 is always slot 42. Deletes are tombstones.
- **Labels are clean.** ≤2048 bytes, no spaces, no `: * ? " < > | ,`. URI-style paths like `docs/file.pdf` are fine.
- **Data is opaque.** ≤100KB per slot. VEC stores the bytes verbatim — sniff the mime on the client if you need to.
- **Same file format across builds.** vec and vec-cpu read/write the same `.tensors`, `.meta`, `.data` files.
- **CRC32 on save.** Pronounceable checksum word for eyeball integrity checks.
- **Read-only mode.** If the file isn't writable, queries work, writes are rejected.
- **Disk space check.** Saves skipped if insufficient space.
- **File repair.** `--check` verifies, `--repair` fixes.

## File format

```
.tensors  [4B dim][4B count][4B deleted][1B fmt][count×1B alive][vectors][4B CRC32]
.meta     [4B count][per slot: 4B len + label bytes]
.data     [4B count][count×1B alive mask][per present slot: 4B len + bytes][4B CRC32]
```

`.data` is new in 2.0 and only created on save when at least one slot has a payload.

## Client SDKs

C++ (`vec_client.h`), Python (`vec_client.py`), Node.js (`vec_client.js`), Delphi (`vec_client.pas`).

All in `sdk/`. Quick protocol reference in `sdk/README.md`. Byte-exact spec in `PROTOCOL-2.0.md`.

## Tested with

DINOv2 (1024d), BGE-large (1024d), ArcFace (512d), MiniLM-L12 (384d).

---

*Curated by [@PsyChip](mailto:root@psychip.net) - April 2026*
