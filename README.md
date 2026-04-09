# VEC

**Dead simple GPU-resident vector database.**

Keep up to 6 million 1024-dim records in your RTX 3090 with a single command. Query them back with ~10ms latency.

Single exe. ~300KB. No libraries. No dependencies. No configuration. Works like memcached - push vectors over TCP, query them back. Except your data lives in GPU VRAM and searches happen inside CUDA cores.

```
vec mydb 1024
```

That's it. Push over TCP, query over TCP. Ctrl+C saves to disk. Designed for NVIDIA CUDA GPUs, currently Windows only. Linux is planned.

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

**VEC** is a TCP/Pipe server that holds vectors in GPU VRAM. Like Redis or memcached, but for vector similarity search.

- Stores and computes in native **fp32** - what you push is exactly what gets searched
- **Two distance metrics**: L2 (Euclidean) and cosine - selectable per query, no rebuild
- **Brute-force search** - no indexing, no approximation, exact results every time
- Returns **top 10 nearest neighbors** per query
- **Undo** last push for quick rollback
- Persists to `.tensors` binary file on disk
- Auto-loads from disk on startup, auto-saves on Ctrl+C
- Supports multiple instances on different ports for different datasets
- No preprocessing, no postprocessing, no normalization - zero data transformation

---

## Installation

### Requirements

- Windows 10/11 (64-bit)
- NVIDIA GPU (Turing or newer)
- NVIDIA display drivers (Game Ready or Studio - that's it, no CUDA Toolkit needed)

### Usage

```cmd
vec <name> <dim> [port]
```

```cmd
vec                        :: uses default values (1024 dimensions, port 1920)
vec mydb 1024              :: default port 1920
vec mydb 1024 1921         :: custom port
vec embeddings 768 1920    :: different db, different dimensions
vec faces 512 1921         :: run multiple instances
```

Each instance gets its own TCP port, named pipe (`\\.\pipe\vec_<name>`), and save file (`<name>.tensors`). Duplicate names are blocked automatically.

---

## Protocol

Plain text over TCP. One command in, one line out. Identical on named pipe.

| Command | Example | Response |
|---|---|---|
| **push** | `push 0.12,0.45,0.78,...\n` | `42\n` (slot index) |
| **pull** | `pull 0.12,0.45,0.78,...\n` | `0:0.0012,5:0.0451,...\n` |
| **cpull** | `cpull 0.12,0.45,0.78,...\n` | `0:0.0012,5:0.0451,...\n` |
| **bpush** | `bpush 1000\n` + raw fp32 bytes | `0\n` (first slot index) |
| **delete** | `delete 42\n` | `ok\n` |
| **undo** | `undo\n` | `ok\n` |
| **save** | `save\n` | `ok\n` |
| **size** | `size\n` | `50\n` |

**push** - send a vector as comma-separated fp32 floats. Returns the slot index. Save this number - it's your key.

**pull** - query by **L2 (Euclidean) distance**. Best for vision models (DINOv2, ArcFace). Measures how far apart two vectors are in space. Cares about both direction and magnitude.

**cpull** - query by **cosine distance**. Best for text embeddings (BGE, MiniLM). Measures the angle between two vectors. Only cares about direction, ignores magnitude. Same data, different metric - no rebuild needed. Unlike FAISS which locks you into one metric at index creation, VEC lets you choose per query.

### L2 vs Cosine

```
  L2 (pull)                          Cosine (cpull)

  How far apart are they?            Are they looking the same way?

       A                                  A ·
      ·                                    /
     /        B                           /        B ·
    /        ·                           /          /
   /  <----->                           /     angle/
  ·  distance ·                        ·----------·
                                         direction

  "Two clouds - we measure             "Two clouds - we measure
   the gap between them"                where they're pointing"

  Sensitive to magnitude:              Ignores magnitude:
  [1,0] vs [10,0] = far               [1,0] vs [10,0] = same direction

  Best for: vision embeddings          Best for: text embeddings
  (DINOv2, ArcFace)                    (BGE, MiniLM, CLIP text)
```

**bpush** - binary bulk push. Send `bpush N\n` followed by `N * dim * 4` bytes of raw fp32 data. ~100x faster than text push.

**delete** - tombstones a slot. Excluded from future queries. Index is preserved, never reused.

**undo** - removes the last pushed vector. The slot index is reclaimed and reused by the next push.

**Errors** return `err <message>\n`:
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
- MSVC Build Tools (v14.41 or older - v14.44 has known cudafe++ crashes)
- Windows SDK

### Build

```cmd
nvcc -O2 -c vec_kernel.cu -o vec_kernel.obj ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89

nvcc -O2 vec_kernel.obj vec.cpp -o vec.exe -lws2_32 ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89
```

| Flag | What it does |
|---|---|
| `sm_75` | Native code for Turing |
| `sm_86` | Native code for Ampere |
| `sm_89` | Native code for Ada Lovelace |

CUDA runtime is statically linked - the output exe needs nothing but NVIDIA display drivers on the target machine.

### Test client

```cmd
cl /O2 /EHsc test.cpp /Fe:test.exe ws2_32.lib
```

```cmd
:: Terminal 1
vec mydb 1024

:: Terminal 2
test mydb 1024
```

Pushes 50 random vectors, queries a noisy copy with both L2 and cosine distance, verifies the original comes back as rank 1.

---

## Internals

**Two source files.** `vec_kernel.cu` has the CUDA kernels, `vec.cpp` has host code. Split is required because CUDA's cudafe++ crashes on Win32/Winsock headers.

**Native fp32.** Vectors stored as fp32 on GPU and on disk. No conversion, no quantization. What you push is what gets searched.

**Grow-on-demand.** GPU buffer starts small, doubles on realloc with device-to-device `cudaMemcpy`. Pinned host memory (`cudaMallocHost`) for fast PCIe transfers.

**Networking.** TCP uses thread-per-client with `select` on the listen socket. Named pipe uses Win32 overlapped I/O. Instance locking via Windows named mutex.

**File format** (`.tensors`):
```
[4B dim][4B count][4B deleted][count B alive mask][count * dim * 4B fp32 data]
```

---

- **Brute force by design.** No ANN index. Every query reads every vector. For up to ~3M vectors on a modern GPU, this is fast enough and gives exact results.
- **Indices are permanent.** Slot 42 stays slot 42 forever. Delete tombstones but never compacts. Your external ID mapping never goes stale.
- **Named pipe works from scripts.** `echo push 1.0,2.0,3.0 > \\.\pipe\vec_mydb` from cmd or PowerShell.
- **Save files are straightforward.** fp32 on disk - a million 1024-dim vectors is ~4 GB.

---

*Curated by [@PsyChip](mailto:root@psychip.net) - April 2026*
