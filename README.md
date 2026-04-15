# VEC

**Dead simple GPU-resident vector database.**

Keep up to 6 million 1024-dim records in your RTX 3090 with a single command. Query them back with ~10ms latency.

Single exe. ~300KB. No libraries. No dependencies. No configuration. Works like memcached - push vectors over TCP, query them back. Except your data lives in GPU VRAM and searches happen inside CUDA cores.

```
vec mydb 1024
```

That's it. Push over TCP, query over TCP. Ctrl+C saves to disk. Designed for NVIDIA CUDA GPUs. Windows and Linux.

---

## Supported GPUs

| Architecture | Generation | Examples |
|---|---|---|
| Turing | RTX 2000 series | RTX 2060, 2070, 2080, T4 |
| Ampere | RTX 3000 series | RTX 3060, 3070, 3080, 3090, A100, A40 |
| Ada Lovelace | RTX 4000 series | RTX 4060, 4070, 4080, 4090, L40 |

Not supported: GTX 1000 series (Pascal) and older. AMD and Intel GPUs are not supported.

## Capacity (fp32 storage)

| Dimensions | 8 GB VRAM | 12 GB VRAM | 24 GB VRAM | 80 GB VRAM |
|---|---|---|---|---|
| 384 | 5.2M | 7.8M | 15.7M | 52M |
| 512 | 3.9M | 5.8M | 11.7M | 39M |
| 768 | 2.6M | 3.9M | 7.8M | 26M |
| 1024 | 1.9M | 2.9M | 5.8M | 19.5M |

## Query latency (RTX 3060, 1024 dim)

| Vectors | Latency |
|---|---|
| 10K | ~0.2 ms |
| 100K | ~1.5 ms |
| 1M | ~14 ms |
| 2.9M | ~40 ms |

Bottleneck is pure memory bandwidth.

---

## How it works

**VEC** is a TCP/pipe server that holds vectors in GPU VRAM. Like Redis or memcached, but for vector similarity search.

- Stores and computes in native **fp32** (fp16 optional for 2x capacity)
- **Two distance metrics**: L2 (Euclidean) and cosine - selectable per query, no rebuild
- **Brute-force search** - no indexing, no approximation, exact results every time
- Returns **top 10 nearest neighbors** per query
- **Labels** - attach metadata to vectors, returned in query results
- **Binary protocol** - bpush/bpull/bcpull for raw fp32 bytes, skips text parsing
- Persists to `.tensors` binary file with CRC32 checksum
- Labels persisted to `.meta` sidecar file
- Auto-loads from disk on startup, auto-saves on Ctrl+C
- Multiple instances on different ports for different datasets
- No preprocessing, no normalization - zero data transformation

---

## Installation

### Requirements

- Windows 10/11 or Linux (64-bit)
- NVIDIA GPU (Turing or newer)
- NVIDIA display drivers (no CUDA Toolkit needed on target)

### Usage

```
vec <name> [dim[:format]] [port]
vec <path/to/file.tensors> [port]
```

```
vec                        :: random name, 1024 dim, fp32, port 1920
vec mydb                   :: auto-detect from existing file or create 1024/fp32
vec mydb 1024              :: fp32 (default)
vec mydb 1024:f16          :: fp16 storage, 2x capacity
vec mydb 1024 1921         :: custom port
vec mydb.tensors           :: load directly from file
```

Each instance gets its own TCP port, named pipe/unix socket, and save file. Duplicate names are blocked automatically.

### File discovery

```
vec mydb                       :: finds mydb_1024_f32.tensors (+ mydb_1024_f32.meta if exists)
vec mydb_1024_f32.tensors      :: loads directly, reads dim/format from file header
vec                            :: random name, fresh database
```

On save, vec writes `<name>_<dim>_<format>.tensors` and optionally `<name>_<dim>_<format>.meta` (only when labels exist). Both files must be in the same directory. If `.meta` is missing, vec loads without labels. If `.tensors` is missing, vec starts a fresh database.

---

## Protocol

Plain text over TCP. One command in, one line out. Identical on named pipe (Windows) and unix socket (Linux).

| Command | Example | Response |
|---|---|---|
| **push** | `push docs/file.pdf?page=2 0.12,0.45,...\n` | `42\n` |
| **push** | `push 0.12,0.45,...\n` | `42\n` (no label) |
| **bpush** | `bpush docs/file.pdf?page=2\n` + raw bytes | `42\n` |
| **bpush** | `bpush\n` + raw bytes | `42\n` (no label) |
| **pull** | `pull 0.12,0.45,...\n` | `docs/file.pdf?page=2:0.0012,...\n` |
| **cpull** | `cpull 0.12,0.45,...\n` | `docs/file.pdf?page=2:0.0012,...\n` |
| **bpull** | `bpull\n` + raw bytes | `docs/file.pdf?page=2:0.0012,...\n` |
| **bcpull** | `bcpull\n` + raw bytes | `docs/file.pdf?page=2:0.0012,...\n` |
| **label** | `label 42 docs/file.pdf?page=2\n` | `ok\n` |
| **delete** | `delete 42\n` | `ok\n` |
| **undo** | `undo\n` | `ok\n` |
| **save** | `save\n` | `ok\n` |
| **size** | `size\n` | `50\n` |

### Push

Send a vector as comma-separated fp32 floats with optional label.

```
push docs/report.pdf?page=2&chunk=4 0.123,0.456,0.789,...\n
-> 42
```

Label is everything before the first float. No label = index-only.

### Binary push

Same as push but vector is raw fp32 bytes. Faster for high-dimensional vectors.

```
bpush docs/report.pdf?page=2\n
<dim * 4 bytes of little-endian fp32 data>
-> 42
```

### Query (pull / cpull / bpull / bcpull)

`pull` = L2 distance, `cpull` = cosine distance. `bpull`/`bcpull` = binary query (raw bytes).

```
pull 0.123,0.456,0.789,...\n
-> docs/report.pdf?page=2:0.001234,photos/beach.jpg:0.045100,42:0.234000
```

Results are `label:distance` pairs (or `index:distance` when no label), comma-separated, nearest first. Up to 10 results.

### L2 vs Cosine

```
  L2 (pull)                          Cosine (cpull)

  How far apart are they?            Are they looking the same way?

       A                                  A .
      .                                    /
     /        B                           /        B .
    /        .                           /          /
   /  <----->                           /     angle/
  .  distance .                        .----------.
                                         direction

  Sensitive to magnitude:              Ignores magnitude:
  [1,0] vs [10,0] = far               [1,0] vs [10,0] = same direction

  Best for: vision embeddings          Best for: text embeddings
  (DINOv2, ArcFace)                    (BGE, MiniLM, CLIP text)
```

### Labels

Labels are metadata strings attached to vectors. They appear in query results instead of numeric indices.

```
label 42 docs/report.pdf?page=2\n
-> ok
```

Labels must not contain colons or commas (stripped with warning). Use URI-style paths without scheme prefix. Labels are saved to a `.meta` sidecar file alongside `.tensors`.

### Errors

All errors start with `err`:
```
err dim mismatch: got 3, expected 1024
err index out of range
err already deleted
err unknown command
```

---

## Tested models

| Model | Dimensions |
|---|---|
| DINOv2 (dinov2_vitl14) | 1024 |
| BGE-large-en-v1.5 | 1024 |
| ArcFace | 512 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 |

All embedding models output fp32. VEC stores fp32 natively - no conversion, no precision loss.

---

## Build from source

### Requirements

- NVIDIA CUDA Toolkit 12.x
- Windows: MSVC Build Tools (v14.41 or older)
- Linux: gcc + nvcc

### Windows

```cmd
build.bat
```

### Linux

```bash
./build.sh
```

### Manual build

```
nvcc -O2 -c vec_kernel.cu -o vec_kernel.obj ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89

nvcc -O2 vec_kernel.obj vec.cpp -o vec.exe -lws2_32 ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89
```

CUDA runtime is statically linked - the output exe needs nothing but NVIDIA display drivers on the target machine.

### Test client

```cmd
cl /O2 /EHsc test.cpp /Fe:test.exe ws2_32.lib
```

```
:: Terminal 1
vec mydb 1024

:: Terminal 2
test mydb 1024
```

Pushes 50 random vectors with labels, queries with L2 and cosine, verifies the original comes back as rank 1.

---

## Internals

**Single source file.** `vec.cpp` has both Windows and Linux code separated by `#ifdef _WIN32`. `vec_kernel.cu` has the CUDA kernels. Split is required because CUDA's cudafe++ crashes on Win32/Winsock headers.

**Vectors on GPU, labels on CPU.** Vectors live in VRAM, labels live in system RAM. The GPU kernel never touches labels - they're stapled onto results after the kernel finishes. Zero performance impact.

**File format** (`.tensors`):
```
[4B dim][4B count][4B deleted][1B format][count B alive mask][vector data][4B CRC32]
```

**Label format** (`.meta` sidecar, created only when labels exist):
```
[4B count][per label: 4B length + string bytes]
```

**CRC32 checksum** on save, verified on load with pronounceable word (e.g. `NOMITOPO 0xA3F291B7`).

---

## Good to know

- **Brute force by design.** No ANN index. Every query reads every vector. For up to ~3M vectors on a modern GPU, this is fast enough and gives exact results.
- **Indices are permanent.** Slot 42 stays slot 42 forever. Delete tombstones but never compacts.
- **Labels are free.** They live in CPU RAM, never touch the GPU. Query performance is identical with or without labels.
- **Labels are sanitized.** Colons and commas are stripped with a console warning. Whitespace is trimmed.
- **Named pipe works from scripts.** `echo push 1.0,2.0,3.0 > \\.\pipe\vec_mydb` from cmd or PowerShell.
- **Save files include CRC32.** Corruption is detected on load with a warning.

---

*Curated by [@PsyChip](mailto:root@psychip.net) - April 2026*
