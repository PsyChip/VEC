# VEC

A dead simple vector database that lives on your GPU. Store millions of embeddings in VRAM. Find the nearest ones in milliseconds. Single exe, no dependencies, no configuration. The entire server (minus CUDA kernels) is ~200 KB of C++ program.

```bash
# Start a database
vec mydb 1024

# That's it. It's listening on port 1920.
```

No GPU? No problem. `vec-cpu` does the same thing using system RAM.

```bash
vec-cpu mydb 1024
```

---

## Protocol

Binary only over Windows named pipe (`\\.\pipe\vec_<name>`) or Linux Unix socket (`/tmp/vec_<name>.sock`). TCP on port 1920 for network access.

```
request:  F0 <2B ns_len> [ns] <CMD> <2B label_len> [label] <4B body_len> [body]
response: <1B status> <4B body_len> [body]
```

16 commands, one envelope. See `PROTOCOL-2.0.md` for the byte-exact spec.

The only way to interact with VEC is through the **SDK packages** — there is no text interface.

| SDK | File |
|-----|------|
| C++ (header-only) | `sdk/vec_client.h` |
| Python 3 (numpy) | `sdk/vec_client.py` |
| Node.js | `sdk/vec_client.js` |
| TypeScript | `sdk/vec_client.d.ts` |
| Delphi | `sdk/vec_client.pas` |

---

## What it does

- **PUSH** — store a vector, optionally with label and ≤100KB data payload.
- **QUERY** — nearest-neighbor search (L2 or cosine).
- **QID** — same as QUERY but by stored vector index or label.
- **EXISTS** — exact-match dedup lookup by vector content.
- **GET** — retrieve records by index, label, or batch.
- **SET_DATA / GET_DATA** — manage per-slot payload independently.
- **UPDATE** — overwrite vector in place (label/data untouched).
- **LABEL / DELETE / UNDO** — manage slots.
- **CLUSTER / DISTINCT / REPRESENT** — DBSCAN, farthest-point sampling, representative selection.
- **INFO / SAVE** — metadata and persistence.

QUERY/QID/GET use a shape mask to control result fields: `0x01` vector, `0x02` label, `0x04` data. Default `0x07` includes all three.

---

## Supported GPUs

| SM | Architecture | GPUs |
|----|-------------|------|
| 7.5 | Turing | RTX 2000 series, T4 |
| 8.0 | Ampere DC | A100 |
| 8.6 | Ampere | RTX 3000 series, A10, A40 |
| 8.9 | Ada Lovelace | RTX 4000 series, L40 |
| 9.0 | Hopper | H100, H200 |
| 10.0 | Blackwell | B100, B200, RTX 5000 series |

Older GPUs are rejected at startup with a clear error. Use `vec-cpu` instead.

---

## Performance

1024-dim vectors:

| Size | GPU (RTX 3060) | CPU |
|------|---------------|-----|
| 10K | ~0.2 ms | ~2 ms |
| 100K | ~1.5 ms | ~20 ms |
| 1M | ~14 ms | ~200 ms |

fp32, 1024-dim capacity:
- 8 GB VRAM → 1.9M vectors
- 12 GB VRAM → 2.9M vectors
- 24 GB VRAM → 5.8M vectors

Lower dimensions or fp16 = more.

---

## File format

```
.tensors  [4B dim][4B count][4B deleted][1B fmt][count×1B alive][vectors][4B CRC32]
.meta     [4B count][per slot: 4B len + label bytes]
.hashes   [4B count][count × 8B u64 xxh64 hash][4B CRC32]
.data     [4B count][count×1B alive mask][per present slot: 4B len + bytes][4B CRC32]
```

`.hashes` is the dedup index — rebuilt from `.tensors` if missing or corrupt. `.data` only written when payloads exist.

---

## Build

Requires NVIDIA CUDA Toolkit 12.x for `vec`. `vec-cpu` just needs a C++ compiler.

```bash
# Windows
build.bat

# Linux
./build.sh
```

---

## Housekeeping

```bash
vec mydb --delete     # destroy database files
vec mydb --check      # verify file integrity
vec mydb --repair     # verify and fix
vec --help            # full reference
```

---

*Created by [@PsyChip](mailto:root@psychip.net) - June 2026*
