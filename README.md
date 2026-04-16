# VEC

**Dead simple GPU-resident vector database.**

Keep up to 6 million 1024-dim records in your RTX 3090. Query them back with ~10ms latency.

Single exe. ~300KB. No libraries. No dependencies. No configuration. Windows and Linux.

```
vec mydb 1024
```

Also ships `vec-cpu` - same protocol, same features, runs on any machine without GPU.

---

## Supported GPUs (vec)

| Architecture | Generation | Examples |
|---|---|---|
| Turing | RTX 2000 series | RTX 2060, 2070, 2080, T4 |
| Ampere | RTX 3000 series | RTX 3060, 3070, 3080, 3090, A100, A40 |
| Ada Lovelace | RTX 4000 series | RTX 4060, 4070, 4080, 4090, L40 |

Not supported: GTX 1000 series (Pascal) and older. AMD and Intel GPUs are not supported. Use `vec-cpu` instead.

## Capacity (fp32)

| Dimensions | 8 GB VRAM | 12 GB VRAM | 24 GB VRAM |
|---|---|---|---|
| 384 | 5.2M | 7.8M | 15.7M |
| 512 | 3.9M | 5.8M | 11.7M |
| 768 | 2.6M | 3.9M | 7.8M |
| 1024 | 1.9M | 2.9M | 5.8M |

## Query latency (1024 dim)

| Vectors | GPU (RTX 3060) | CPU |
|---|---|---|
| 10K | ~0.2 ms | ~3 ms |
| 100K | ~1.5 ms | ~30 ms |
| 1M | ~14 ms | ~300 ms |

---

## Usage

```
vec <name> [dim[:format]] [port]
vec <file.tensors> [port]
vec --help
vec --notcp <name> [dim] [port]
vec --route <port>
```

```
vec                        :: auto-detect .tensors in dir or create new
vec mydb                   :: find mydb_*.tensors or create 1024/fp32
vec mydb 1024              :: explicit dimension
vec mydb 1024:f16          :: fp16 storage, 2x capacity
vec mydb 1024 1921         :: custom port
vec mydb.tensors           :: load from file
vec --help                 :: show all commands
```

### File discovery

On startup with no arguments, vec scans the current directory for `.tensors` files and loads the best match. When loading, also loads `.meta` file for labels if it exists.

### Router mode

Run multiple instances with pipe-only mode, route them through one TCP port:

```
vec --notcp tools 1024
vec --notcp conversations 1024
vec --route 1920
```

Client sends namespace-prefixed commands: `tools push 0.12,...\n`

---

## Protocol

Plain text over TCP. One command in, one line out. Same on named pipe (Windows) and unix socket (Linux).

| Command | Example | Response |
|---|---|---|
| **push** | `push label 0.12,0.45,...\n` | `42\n` |
| **push** | `push "quoted label" 0.12,0.45,...\n` | `42\n` |
| **bpush** | `bpush label\n` + raw bytes | `42\n` |
| **pull** | `pull 0.12,0.45,...\n` | `label:0.0012,...\n` |
| **cpull** | `cpull 0.12,0.45,...\n` | `label:0.0012,...\n` |
| **bpull** | `bpull\n` + raw bytes | `label:0.0012,...\n` |
| **bcpull** | `bcpull\n` + raw bytes | `label:0.0012,...\n` |
| **label** | `label 42 "new label"\n` | `ok\n` |
| **delete** | `delete 42\n` | `ok\n` |
| **undo** | `undo\n` | `ok\n` |
| **save** | `save\n` | `ok\n` |
| **size** | `size\n` | `50\n` |

### Push

```
push docs/report.pdf?page=2 0.12,0.45,...\n          -> 42
push "user: how does the engine work?" 0.12,0.45,...\n -> 43
push 0.12,0.45,...\n                                    -> 44 (no label)
```

Use quotes for labels with spaces. No quotes needed for simple paths.

### Binary push/query

```
bpush label\n<dim*4 bytes>     -> 42 (binary push)
bpull\n<dim*4 bytes>           -> results (L2 binary query)
bcpull\n<dim*4 bytes>          -> results (cosine binary query)
```

Same as text versions but vector sent as raw fp32 bytes. Skips CSV parsing.

### Query results

```
pull 0.12,0.45,...\n
-> docs/report.pdf?page=2:0.001234,photos/beach.jpg:0.045100,42:0.234000
```

`label:distance` pairs (or `index:distance` when no label). Comma-separated, nearest first. Up to 10 results. Parse by splitting on last colon per result.

### L2 vs Cosine

```
  L2 (pull/bpull)                    Cosine (cpull/bcpull)
  How far apart?                     Looking the same way?
  [1,0] vs [10,0] = far             [1,0] vs [10,0] = same direction
  Vision: DINOv2, ArcFace            Text: BGE, MiniLM, CLIP
```

### Labels

Labels must not contain colons or commas (stripped with warning). Newlines and tabs escaped to `\n` and `\t`. UTF-8 BOM stripped. Use quotes for labels with spaces.

```
label 42 docs/report.pdf?page=2\n        -> ok
label 42 "label with spaces"\n           -> ok
```

### Errors

```
err dim mismatch: got 3, expected 1024
err index out of range
err already deleted
err unknown command
```

---

## vec-cpu

Same protocol, same file format, same features. Runs on any machine without GPU.

```
vec-cpu mydb 1024
```

### Performance (CPU, 1024 dim)

| Records | Latency |
|---|---|
| 10K | ~3 ms |
| 100K | ~30 ms |
| 1M | ~300 ms |

### Requirements

- Windows or Linux (64-bit)
- No GPU needed, no CUDA needed
- fp32 only (fp16 requires GPU)

On startup, checks available RAM. Warns if less than 8 GB.

---

## Build

### Requirements

- `vec`: NVIDIA CUDA Toolkit 12.x, MSVC (v14.41 or older) or gcc + nvcc
- `vec-cpu`: just MSVC or gcc (no CUDA needed)

### Windows

```cmd
build.bat
```

Builds: `vec.exe`, `vec-cpu.exe`, `test.exe`, `test_route.exe`

### Linux

```bash
./build.sh
```

Builds: `vec`, `vec-cpu`

### Manual build

```
:: vec (GPU)
nvcc -O2 -c vec_kernel.cu -o vec_kernel.obj <gencode flags>
nvcc -O2 vec_kernel.obj vec.cpp -o vec.exe -lws2_32 -lmpr <gencode flags>

:: vec-cpu (CPU)
cl /O2 /EHsc vec-cpu.cpp /Fe:vec-cpu.exe ws2_32.lib mpr.lib

:: test
cl /O2 /EHsc test.cpp /Fe:test.exe ws2_32.lib
cl /O2 /EHsc test_route.cpp /Fe:test_route.exe ws2_32.lib
```

### Test

```
:: single instance test
vec mydb 1024
test mydb 1024

:: router test (5 namespaces)
start vec --notcp tools 3
start vec --notcp conversations 3
start vec --notcp faces 3
start vec --notcp clips 3
start vec --notcp notes 3
start vec --route 1920
test_route 1920
```

---

## Startup

On startup, vec prints a status block:

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

vec-cpu shows RAM instead of GPU:

```
vec-cpu (32.0 GB RAM)
===================================================================
  ...
===================================================================
```

---

## File format

### .tensors

```
[4B dim][4B count][4B deleted][1B format][count B alive mask][vector data][4B CRC32]
```

Naming: `<name>_<dim>_<format>.tensors`

### .meta (labels, only when labels exist)

```
[4B count][per label: 4B length + string bytes]
```

---

## Good to know

- **Brute force by design.** Every query reads every vector. Exact results, no approximation.
- **Indices are permanent.** Slot 42 stays 42 forever. Delete tombstones but never compacts.
- **Labels are free.** CPU RAM only. Zero impact on query performance.
- **Labels are sanitized.** Colons/commas stripped, newlines/tabs escaped, BOM removed, whitespace trimmed.
- **Quoted labels.** Use `"quotes"` for labels with spaces: `push "hello world" 0.12,...`
- **CRC32 on save.** Pronounceable checksum word for quick visual verification.
- **Same file format.** vec and vec-cpu read/write the same .tensors and .meta files.

---

## Tested models

| Model | Dimensions | Use |
|---|---|---|
| DINOv2 (dinov2_vitl14) | 1024 | Image embeddings |
| BGE-large-en-v1.5 | 1024 | Text embeddings |
| ArcFace | 512 | Face recognition |
| MiniLM-L12-v2 | 384 | Multilingual text |

---

## Client SDKs

Available in `sdk/` directory:

- `vec_client.h` - C++
- `vec_client.py` - Python
- `vec_client.js` - Node.js
- `vec_client.pas` - Delphi

All implement: `push`, `bpush`, `pull`, `cpull`, `bpull`, `bcpull`, `setLabel`, `delete`, `undo`, `save`, `size`.

---

*Curated by [@PsyChip](mailto:root@psychip.net) - April 2026*
