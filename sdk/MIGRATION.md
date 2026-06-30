# VEC 2.0 Migration Guide

VEC 2.0 is a clean break from 1.x. The protocol is incompatible at the wire level — old SDKs cannot speak to a 2.0 server, and new SDKs cannot speak to a 1.x server. **You must update server and clients together.**

If you need the byte-exact spec, see `PROTOCOL-2.0.md`. Quick command reference is in `sdk/README.md`.

---

## Why 2.0

The MySQL sidekick is dead. 1.x stored vectors and labels in VEC and the actual document/payload bytes in MySQL or another keyed store. 2.0 adds a **data field** so each record is a self-contained `(vector, label, ≤100KB blob)` tuple — VEC alone replaces the vec+MySQL pair for most use cases.

To make that work cleanly the protocol was rebuilt:

- Every request now carries an explicit `4B body_len` after the header.
- Every response is a binary envelope `[1B status][4B body_len][body]` — no more `ok\n` / `err ...\n` text.
- Search and fetch commands gained a **shape mask** so callers pick what each record carries (vector and/or label and/or data).
- Redundant commands were merged: PULL+CPULL → QUERY, PID+CPID → QID, GET+MGET → GET.
- Labels are now hard-validated (no spaces, no `: * ? " < > | ,`, no control chars) and capped at 2048 bytes. URI-style labels (`docs/file.pdf`) are still fine.

---

## What stays compatible

- **`.tensors` and `.meta` files are unchanged.** A 1.x DB loads fine into a 2.0 server.
- **Existing labels are loaded leniently** — labels that violate the new filename scheme stay in the DB until rewritten. Only *new* writes (PUSH, CMD_LABEL) enforce validation.
- The build, transport options (TCP/named pipe/Unix socket), the router, and the deploy/multi-DB modes are unchanged.

## What's a breaking change

- **Protocol wire format**: every frame and every response.
- **Removed commands**: `0x03` (CPULL), `0x05` (MGET), `0x12` (CPID).
- **Renamed commands**: `0x02` is now QUERY, `0x11` is now QID.
- **New commands**: `0x13` (SET_DATA), `0x14` (GET_DATA).
- **DELETE / UNDO** now also clear the slot's label and data alongside the vector.
- **PUSH** signature: data is optional but **requires a label** when present.
- **Response shapes** for QUERY/QID/GET are length-prefixed binary records, not `label:dist` text.
- **CLUSTER / DISTINCT / REPRESENT** responses are now wrapped in the binary envelope; their bodies remain legacy text (lines + `end`).

---

## Step 1 — Update server

Drop in the new `vec.cpp` / `vec-cpu.cpp` / `vec_kernel.cu` and rebuild:

```bash
./build.sh         # Linux
./build.bat        # Windows
```

The server reads existing `.tensors` and `.meta` files unchanged. On first SAVE after you start using SET_DATA, a new `.data` sidecar is created next to them.

---

## Step 2 — Replace SDKs

The SDKs in `sdk/` are 2.0-only. Drop in the new files; do not mix versions.

| File | Status |
|------|--------|
| `sdk/vec_client.h` | rewritten |
| `sdk/vec_client.py` | rewritten |
| `sdk/vec_client.js` | rewritten |
| `sdk/vec_client.pas` | rewritten |

The 2.0 API is similar enough to 1.x that most call sites need only small edits — see Step 3.

---

## Step 3 — Update call sites

### Python

**Before (1.x):**
```python
from vec_client import VecClient

vec = VecClient("localhost", 1920)
idx = vec.push([0.1, 0.2], label="doc1")
results = vec.pull([0.1, 0.2])           # → [VecResult(label, distance)]
results = vec.cpull([0.1, 0.2])
results = vec.pid(42)
results = vec.cpid("doc1")
arrays = vec.get(42)                      # → [np.ndarray]
arrays = vec.mget([0, 1, 2])
vec.update(42, [0.1, 0.2])
vec.setLabel(42, "doc1")
vec.delete(42)
info = vec.info()                          # dict
```

**After (2.0):**
```python
from vec_client import VecClient, SHAPE_LABEL, SHAPE_FULL

vec = VecClient("localhost", 1920)

# push gained an optional data parameter; data requires label
idx = vec.push([0.1, 0.2], label="doc1")
idx = vec.push([0.1, 0.2], label="img/cat.jpg", data=open("cat.jpg","rb").read())

# pull/cpull → query(cosine=...)
results = vec.query([0.1, 0.2])                       # full record default
results = vec.query([0.1, 0.2], cosine=True)
results = vec.query([0.1, 0.2], shape=SHAPE_LABEL)    # lean (no vector, no data)

# pid/cpid → qid(cosine=...)
results = vec.qid(42)
results = vec.qid("doc1", cosine=True)

# get/mget unified; takes int, str, or list[int]
recs = vec.get(42)
recs = vec.get("doc1")        # may return multiple if ambiguous
recs = vec.get([0, 1, 2])     # batch

# new sidecar payload management
vec.set_data(42, b"...binary...")
blob = vec.get_data(42)

vec.update(42, [0.1, 0.2])    # vector only
vec.set_label(42, "doc1")
vec.delete(42)                 # also clears label + data
info = vec.info()              # now includes 'protocol' key
```

**Result types changed.** Old `pull` returned `VecResult(index, distance, label)`. New `query` returns `VecRecord(index, distance, label, data, vector)` — fields that weren't requested via shape are `None`. Old `get` returned a list of `np.ndarray`; new `get` returns a list of `VecRecord` (vector under `.vector`).

### Node.js

**Before (1.x):**
```javascript
const VecClient = require('./vec_client');
const vec = new VecClient('localhost', 1920);
await vec.connect();

const idx = await vec.push([0.1, 0.2], 'doc1');
const r = await vec.pull([0.1, 0.2]);    // [{ index, distance, label }]
const r2 = await vec.cpull([0.1, 0.2]);
const r3 = await vec.pid(42);
const r4 = await vec.cpid('doc1');
const arr = await vec.get('doc1', 1024);  // Float32Array[]
const batch = await vec.mget([0,1,2], 1024);
await vec.update(42, [0.1, 0.2]);
const info = await vec.info();
```

**After (2.0):**
```javascript
const VecClient = require('./vec_client');
const vec = new VecClient('localhost', 1920);
await vec.connect();

// push: third arg is now an options object with optional label+data
const idx = await vec.push([0.1, 0.2], { label: 'doc1' });
const idx2 = await vec.push([0.1, 0.2], { label: 'img/cat.jpg', data: jpegBuffer });

// query / qid (replaces pull/cpull/pid/cpid)
const r1 = await vec.query([0.1, 0.2]);                       // full
const r2 = await vec.query([0.1, 0.2], { cosine: true });
const r3 = await vec.query([0.1, 0.2], { shape: VecClient.SHAPE_LABEL });
const r4 = await vec.qid(42);
const r5 = await vec.qid('doc1', { cosine: true });

// get unified
const a = await vec.get(42);
const b = await vec.get('doc1');
const c = await vec.get([0, 1, 2]);

// new payload management
await vec.setData(42, jpegBuffer);
const blob = await vec.getData(42);

await vec.update(42, [0.1, 0.2]);
const { savedCount, crc } = await vec.save();
const info = await vec.info();    // info.protocol === 2
```

Records now carry `{ index, distance, label, data, vector }` — `null` for fields not in the shape mask. `vector` is a `Float32Array` when included.

### C++

**Before (1.x):**
```cpp
VecClient vec;
vec.connect_tcp("localhost", 1920);

int idx = vec.push("doc1", v, dim);
VecResult rs[10];
int n = vec.pull(v, dim, rs, 10);
int n2 = vec.cpull(v, dim, rs, 10);
int n3 = vec.pid(42, rs, 10);
int n4 = vec.cpid("doc1", rs, 10);

float out[1024];
int got = vec.get(42, out, 1024);          // floats written
int got2 = vec.get("doc1", out, 1024);
int got3 = vec.mget(idx_arr, 3, out, 1024);

vec.update(42, v, dim);
vec.setLabel(42, "doc1");
vec.vec_delete(42);
VecInfo info; vec.info(&info);
```

**After (2.0):**
```cpp
VecClient vec;
vec.connect_tcp("localhost", 1920);

int idx = vec.push("doc1", v, dim);
int idx2 = vec.push("img/cat.jpg", v, dim, jpeg_bytes, jpeg_len); // with data

// query: returns VecRecord array; each record owns malloc'd buffers
VecRecord rs[10];
int n = vec.query(v, dim, rs, 10);                                  // full record default
int n2 = vec.query(v, dim, rs, 10, /*cosine*/1, VEC_SHAPE_LABEL);    // cosine, lean
int n3 = vec.qid(42, rs, 10);
int n4 = vec.qid("doc1", rs, 10, 1);                                  // cosine

int n5 = vec.get(42, rs, 10);                          // single by index
int n6 = vec.get("doc1", rs, 10);                       // by label (may be multi)
int n7 = vec.get_batch(idx_arr, 3, rs, 10);             // batch

// caller frees per-record buffers
for (int i = 0; i < n; i++) vec_free_record(&rs[i]);

vec.set_data(42, jpeg_bytes, jpeg_len);
unsigned char *blob; unsigned int blob_len;
vec.get_data(42, &blob, &blob_len);
free(blob);

vec.update(42, v, dim);
vec.set_label(42, "doc1");
vec.delete_index(42);                                   // was vec_delete
unsigned int saved, crc; vec.save(&saved, &crc);
VecInfo info; vec.info(&info);                          // info.protocol_version == 2
```

Method renames: `pull` → `query` (with cosine flag), `pid` → `qid`, `cpull`/`cpid` removed (use cosine flag), `mget` → `get_batch`, `vec_delete` → `delete_index`.

`VecResult` (1.x: index/distance/label only) is replaced by `VecRecord` which also carries `data`, `vector`, and length fields. Each record's `label`/`data`/`vector` is malloc'd by the SDK — call `vec_free_record(&r)` on each before reusing or scope-exit.

### Delphi

| 1.x | 2.0 |
|-----|-----|
| `Pull(v)` | `Query(v)` |
| `CPull(v)` | `Query(v, True)` |
| `PID(idx)` | `QID(idx)` |
| `PIDByLabel(s)` | `QIDByLabel(s)` |
| `CPID(idx)` | `QID(idx, True)` |
| `CPIDByLabel(s)` | `QIDByLabel(s, True)` |
| `Get(idx, dim)` → `TSingleArray` | `Get(idx)` → `TVecRecords` |
| `GetByLabel(s, dim)` | `GetByLabel(s)` |
| `MGet(arr, dim)` | `GetBatch(arr)` |
| — | `PushFull(label, vec, data)` |
| — | `SetData(idx, data)` / `GetData(idx)` |
| — | `Save(out saved, out crc)` (returns counters) |
| `Info` returns `TVecInfo` | same, plus `ProtocolVersion: Integer` |

`TVecResult` is replaced by `TVecRecord` (Index, Distance, Label_, Data, Vector).

---

## Step 4 — Adjust to new behaviors

### DELETE clears label and data

In 1.x, DELETE only flipped the alive bit. In 2.0, DELETE also frees the slot's label and data. If you relied on a deleted slot retaining its label so you could re-LABEL it before re-pushing, you'll need to capture it before delete.

### UNDO clears last label and data

Same change applied to UNDO.

### Label validation on writes

PUSH and CMD_LABEL now reject:
- length > 2048
- spaces, tabs, control chars
- any of `: * ? " < > | ,`

If your existing code generated labels with these characters, the writes will start failing with `label has invalid chars`. Existing labels in `.meta` files are kept as-is (lenient load).

### PUSH data requires label

A PUSH with a non-empty data field but no label returns `data requires label`. There is no anonymous-blob path — every payload is addressable by its label.

### Default response shape is full

`query` / `qid` / `get` default to `SHAPE_FULL` (`0x07` = vector + label + data). If you don't want every record to carry up to ~100KB of data, pass a leaner shape:

- `SHAPE_VECTOR` (`0x01`): vector only — closest to 1.x behavior
- `SHAPE_LABEL` (`0x02`): index + label only — for fast hit-list lookups
- `SHAPE_VECTOR | SHAPE_LABEL` (`0x03`): vector + label

### Removed text protocol

There is no `ok\n` / `err ...\n` line on the wire anymore. Every response is `[1B status][4B body_len][body]`. If you had any residual code parsing line-based responses (e.g. raw `nc` health checks), update them — see Step 5.

---

## Step 5 — Update health checks and monitoring

A 1.x INFO probe could read fixed-26 bytes plus a name. In 2.0 you now read the response envelope first.

**Python health check:**
```python
from vec_client import VecClient
try:
    vec = VecClient("localhost", 1920)
    info = vec.info()
    print(f"ok protocol={info['protocol']} count={info['count']} dim={info['dim']}")
    vec.close()
except Exception as e:
    print(f"fail: {e}")
    raise SystemExit(1)
```

**Raw INFO probe (frame to send):**
```
F0 00 00 10 00 00 00 00 00 00
^   ^----^      ^----^ ^---------^
|   ns_len=0    lbl=0  body_len=0
|       (then CMD_INFO=10)
magic
```

Server replies with the response envelope — `01` byte = error, `00` byte = ok followed by 4B body_len + body.

---

## Quick reference

### 1.x → 2.0 command codes

| 1.x | 2.0 | Notes |
|-----|-----|-------|
| `0x01` PUSH | `0x01` PUSH | body now `vec + 4B dlen + data` |
| `0x02` PULL | `0x02` QUERY | metric byte selects L2/cosine |
| `0x03` CPULL | — | use QUERY with cosine=1 |
| `0x04` GET | `0x04` GET | mode byte (single/batch) + shape byte |
| `0x05` MGET | — | use GET batch mode |
| `0x06` UPDATE | `0x06` UPDATE | unchanged |
| `0x07` DELETE | `0x07` DELETE | also clears label + data |
| `0x08` LABEL | `0x08` LABEL | unchanged |
| `0x09` UNDO | `0x09` UNDO | also clears last label + data |
| `0x0A` SAVE | `0x0A` SAVE | response is `<4B saved><4B crc>` |
| `0x0D` CLUSTER | `0x0D` CLUSTER | response wrapped in binary envelope (body still text) |
| `0x0E` DISTINCT | `0x0E` DISTINCT | same |
| `0x0F` REPRESENT | `0x0F` REPRESENT | same |
| `0x10` INFO | `0x10` INFO | adds 1B protocol version at end |
| `0x11` PID | `0x11` QID | metric byte |
| `0x12` CPID | — | use QID with cosine=1 |
| — | `0x13` SET_DATA | new |
| — | `0x14` GET_DATA | new |

### 1.x → 2.0 SDK method renames

| 1.x | 2.0 |
|-----|-----|
| `pull(v)` | `query(v)` |
| `cpull(v)` | `query(v, cosine=True)` |
| `pid(x)` | `qid(x)` |
| `cpid(x)` | `qid(x, cosine=True)` |
| `mget(idx_arr)` (Python) | `get(idx_arr)` |
| `mget(idx_arr, dim)` (JS) | `get(idx_arr)` |
| `MGet(idx_arr)` (Pascal) | `GetBatch(idx_arr)` |
| `mget(int*, n, ..., dim)` (C++) | `get_batch(int*, n, rs, max)` |
| `vec_delete(idx)` (C++) | `delete_index(idx)` |
| `setLabel(...)` | `set_label(...)` (Python/C++) |
| — | `set_data` / `get_data` (all clients) |
