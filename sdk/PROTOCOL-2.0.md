# VEC 2.0 Wire Protocol

All inputs and outputs are binary, length-prefixed. SDKs from 1.x will not work.

---

## 1. Frame envelope

### Request (client → server)

```
F0  <2B ns_len>  [ns_bytes]  <CMD>  <2B label_len>  [label_bytes]  <4B body_len>  [body_bytes]
```

- `F0` (1B) — magic byte `BIN_MAGIC`.
- `ns_len` (2B LE u16) — namespace name length. `0` = direct mode (no router). Used by deploy/router builds.
- `ns_bytes` — namespace name, ASCII, no NUL. Present only when `ns_len > 0`.
- `CMD` (1B) — command code. See §3.
- `label_len` (2B LE u16) — label length. `0` = no label. Hard cap: `2048`.
- `label_bytes` — UTF-8 label (no NUL terminator on wire). Present only when `label_len > 0`.
- `body_len` (4B LE u32) — length of `body_bytes`. **New in 2.0** — every command carries an explicit body length, replacing the per-command inferred length table from 1.x.
- `body_bytes` — command-specific payload. See §3 for layout per command.

> **Why an explicit body_len?** PUSH and SET_DATA carry variable-length blobs; QUERY/QID/GET carry option bytes plus optional payloads. A single `body_len` field removes all per-command size inference and lets the parser advance cleanly.

### Response (server → client)

```
<1B status>  <4B body_len>  [body_bytes]
```

- `status` (1B) — `0x00` = OK, `0x01` = ERR.
- `body_len` (4B LE u32) — length of `body_bytes`.
- `body_bytes` — on OK: command-specific response (see §3). On ERR: ASCII error text, no trailing newline.

> **Symmetric**: every response uses this envelope. There is no `ok\n` / `err ...\n` text protocol in 2.0.

---

## 2. Common types

| Symbol | Encoding | Notes |
|---|---|---|
| `u8` | 1B | |
| `u16` | 2B little-endian | |
| `u32` | 4B little-endian | |
| `i32` | 4B little-endian, two's complement | |
| `i64` | 8B little-endian, two's complement | |
| `f32` | 4B IEEE-754 little-endian | |
| `vec` | `dim × f32` | Vector payload. `dim` is fixed per-DB. |
| `varbytes` | `<4B u32 len>[len bytes]` | Length-prefixed blob. |

### Shape mask (1B u8)

Used by QUERY, QID, GET to control what each result record carries:

| Bit | Meaning | Cost |
|---|---|---|
| `0` (`0x01`) | include vector | `dim × 4` bytes per record |
| `1` (`0x02`) | include label | `varbytes` per record (≤2052 B) |
| `2` (`0x04`) | include data | `varbytes` per record (≤102404 B) |
| `3..7` | reserved, must be zero | — |

**Default** (recommended client value): `0x07` (vector + label + data — full record, replaces the 1.x MySQL-sidecar workflow).

A shape mask of `0x00` is legal but useless (only index/distance fields, no payload).

### Metric byte (1B u8)

| Value | Meaning |
|---|---|
| `0x00` | L2 (Euclidean²) — **default** |
| `0x01` | cosine (1 − cos similarity) |
| others | reserved, ERR |

### Mode byte (1B u8) for GET

| Value | Meaning |
|---|---|
| `0x00` | single — body carries one `i32` index (or label resolution via header `label_len > 0`) |
| `0x01` | batch — body carries `<4B u32 count>[count × i32]` indices |
| others | reserved, ERR |

### Mode byte (1B u8) for CLUSTER / DISTINCT / REPRESENT

| Value | Meaning |
|---|---|
| `0x00` | L2 |
| `0x01` | cosine |

(Same metric semantics as QUERY but separate field — these commands existed pre-merge.)

---

## 3. Commands

### `0x01` PUSH

Insert a new record. Vector required. Label optional. **Data requires a label** — a record cannot have a payload without an addressable name.

**Request body**
```
vec                          ; dim × f32, REQUIRED
<4B u32 data_len>            ; 0 = no data; >0 requires header label_len > 0
[data_len bytes]             ; opaque blob, ≤102400 bytes
```

Label is supplied via the **header** `label_len` / `label_bytes` (consistent with all other label-carrying commands).

**Response body (OK)**
```
<4B i32 index>
```

**Errors**
- body too short / vector incomplete
- label invalid (filename validation, see §4)
- label_len > 2048
- data_len > 102400
- `data_len > 0` with `label_len == 0` → ERR `data requires label`
- vector already exists in DB → ERR `duplicate vector at index <N>`

**Deduplication.** Every PUSH is hashed (xxh64 over stored bytes) and byte-compared against alive slots. A bit-identical vector is rejected with the index of the existing record. Tombstoned slots do not block dedup. UPDATE overwrites without a dedup check; DELETE/UNDO remove the entry.

---

### `0x02` QUERY (replaces 1.x PULL + CPULL)

Nearest-neighbor search by query vector.

**Request body**
```
<1B metric>                  ; 0 = L2, 1 = cosine
<1B shape>                   ; bitmask
vec                          ; dim × f32 query
```

**Response body (OK)**
```
<4B u32 count>               ; number of results (top-K)
for each result:
  <4B i32 index>
  <4B f32 distance>
  if shape & 0x02:  <4B u32 lbl_len>[lbl_len bytes]
  if shape & 0x04:  <4B u32 data_len>[data_len bytes]
  if shape & 0x01:  vec      ; dim × f32
```

Top-K size: GPU build = 16 (`GPU_TOP_K`, hardcoded). CPU build defaults to 50, adjustable via `--topk=N` flag (max 100).

---

### `0x04` GET (replaces 1.x GET + MGET)

Fetch one or many records by index, or one record by label.

**Request body**
```
<1B mode>                    ; 0 = single, 1 = batch
<1B shape>                   ; bitmask
if mode == 0:
  if header label_len > 0:   ; label resolution path — body has nothing else
    (empty)
  else:
    <4B i32 index>
if mode == 1:
  <4B u32 count>
  count × <4B i32 index>
```

**Response body (OK)**
```
<4B u32 count>
for each result:
  <4B i32 index>
  if shape & 0x02:  <4B u32 lbl_len>[lbl_len bytes]
  if shape & 0x04:  <4B u32 data_len>[data_len bytes]
  if shape & 0x01:  vec
```

Single mode with a label that resolves to multiple matches returns **all matches** (same as 1.x `resolve_label_all`).

---

### `0x06` UPDATE

Overwrite the vector at a slot. Label and data are **not** modified by UPDATE — use CMD_LABEL or SET_DATA for those.

**Request body**
```
if header label_len > 0:
  vec                        ; dim × f32
else:
  <4B i32 index>
  vec
```

**Response body (OK)**: empty.

---

### `0x07` DELETE

Tombstone a slot. Clears the alive bit, frees the label, frees the data. The vector slot remains addressable as deleted (consistent with 1.x).

**Request body**
```
if header label_len > 0:
  (empty)
else:
  <4B i32 index>
```

**Response body (OK)**: empty.

---

### `0x08` CMD_LABEL

Set or clear the label for an existing slot.

**Request body**
```
<4B i32 index>
```

The label is supplied via the **header** `label_len` / `label_bytes` (consistent with 1.x). `label_len = 0` clears.

**Response body (OK)**: empty.

**Errors**
- label invalid (filename validation, see §4)
- label_len > 2048

---

### `0x09` UNDO

Remove the last PUSH. Frees the label and data of the popped slot.

**Request body**: empty.
**Response body (OK)**: empty.

---

### `0x0A` SAVE

Flush `.tensors`, `.meta`, `.data` to disk.

**Request body**: empty.
**Response body (OK)**
```
<4B u32 saved_count>
<4B u32 crc32>
```

---

### `0x0D` CLUSTER (DBSCAN)

**Request body**
```
<4B f32 eps>
<1B mode>                    ; 0 = L2, 1 = cosine
<4B i32 min_pts>
```

**Response body (OK)**
```
<4B u32 cluster_count>
for each cluster:
  <4B u32 member_count>
  <member_count × 4B i32 index>
  <dim × 4B f32 centroid>    ; fp32 on the wire even when DB is f16
<4B u32 noise_count>
<noise_count × 4B i32 index>
```

The centroid is the unweighted arithmetic mean of the cluster members (computed
fp32-host-side even for f16 databases). Use it to detect degenerate-attractor
clusters: clusters whose centroid magnitude is suspiciously small, or whose
intra-cluster pairwise distance is large compared to other clusters, often
contain low-confidence/fallback embeddings rather than a real coherent group.

Noise points are reported only as indices — no centroid.

---

### `0x0E` DISTINCT (farthest-point sampling)

**Request body**
```
<4B i32 k>
<1B mode>
```

**Response body (OK)** — legacy text, one result per line, terminated with `end\n`:
```
<index_or_label>\n
...
end\n
```

---

### `0x0F` REPRESENT (one rep per DBSCAN cluster)

**Request body**
```
<4B f32 eps>
<1B mode>
<4B i32 min_pts>
```

**Response body (OK)** — legacy text, one result per line, terminated with `end\n`:
```
<index_or_label>\n
...
end\n
```

---

### `0x10` INFO

**Request body**: empty.

**Response body (OK)**
```
<4B i32 dim>
<4B i32 count>
<4B i32 deleted>
<1B  u8  fmt>                ; 0 = f32, 1 = f16
<8B  i64 mtime>              ; unix epoch seconds
<4B u32 crc32>
<1B  u8  crc_ok>             ; 0 = mismatch, 1 = ok, 2 = unknown (no file yet)
<4B u32 name_len>
<name_len bytes>             ; DB name
<1B u8 protocol_version>     ; new in 2.0: always 0x02
```

---

### `0x11` QID (replaces 1.x PID + CPID)

Nearest-neighbor search using a stored vector as query.

**Request body**
```
<1B metric>
<1B shape>
if header label_len > 0:
  (empty)
else:
  <4B i32 index>
```

**Response body (OK)**: identical to QUERY (§ `0x02`).

---

### `0x13` SET_DATA — **NEW**

Set or clear the blob payload for a slot.

**Request body**
```
if header label_len > 0:
  <4B u32 data_len>
  [data_len bytes]
else:
  <4B i32 index>
  <4B u32 data_len>
  [data_len bytes]
```

`data_len = 0` clears the blob. `data_len > 102400` → ERR.

**Response body (OK)**: empty.

---

### `0x14` GET_DATA — **NEW**

Fetch the blob payload for a slot.

**Request body**
```
if header label_len > 0:
  (empty)
else:
  <4B i32 index>
```

**Response body (OK)**
```
<4B u32 data_len>
[data_len bytes]
```

`data_len = 0` indicates the slot has no blob (not an error).

---

### `0x15` EXISTS — **NEW**

Look up a record by exact vector content. Used to check for duplicates before PUSH or to fetch the index/label/data of an existing record without a similarity scan.

**Request body**
```
<1B shape>                   ; bitmask: 0x02=label, 0x04=data; 0x01 ignored
vec                          ; dim × f32
```

**Response body (OK)**
```
<1B u8 found>                ; 0 = no match, 1 = match
if found:
  <4B i32 index>             ; always present
  if shape & 0x02: <4B u32 lbl_len>[lbl_len bytes]
  if shape & 0x04: <4B u32 data_len>[data_len bytes]
```

`shape & 0x01` (vector) is meaningless — caller already supplied the vector — and is ignored. Tombstoned slots are skipped. Match is byte-exact against the stored representation (so for f16 DBs, the input is converted lane-by-lane to binary16 before comparison).

Backed by an in-memory `xxh64 → slot` index, persisted in the `.hashes` sidecar.

---

### Reserved / removed

| Code | Status |
|---|---|
| `0x03` | **removed** (was CPULL; merged into QUERY) |
| `0x05` | **removed** (was MGET; merged into GET) |
| `0x0B`, `0x0C` | reserved (do not reuse) |
| `0x12` | **removed** (was CPID; merged into QID) |

---

## 4. Validation rules

### Labels (PUSH, CMD_LABEL)

Hard-rejected on **write**:
- length > 2048 bytes
- contains any of: space, tab, control chars (`\x00`–`\x1F`, `\x7F`), `:`, `*`, `?`, `"`, `<`, `>`, `|`, `,`
- empty after BOM/whitespace strip (existing 1.x sanitization)

Allowed: alphanumerics, `/`, `\`, `.`, `_`, `-`, `=`, `!`, `'`, `^`, `+`, `%`, `&`, `(`, `)`, plus other printable punctuation not in the rejected set. URI-style labels like `docs/file.pdf` are fine.

Validation is enforced **only on writes**. Existing `.meta` files load with 1.x's lenient rules — pre-2.0 labels that violate the new scheme are kept as-is until rewritten.

### Data (PUSH, SET_DATA)

- Hard cap: 102400 bytes (100 KB).
- Opaque bytes — no encoding validation, no sniffing.
- `data_len = 0` is valid and means "no blob" (clears existing if any).

### Vectors (PUSH, UPDATE, QUERY)

- Must be exactly `dim × 4` bytes (f32 wire format).
- Server-side fmt conversion to f16 happens in `upload_and_store` if DB is f16 — wire format stays f32 either way.

---

## 5. Storage layout

### `.tensors` — unchanged from 1.x
```
<4B i32 dim>
<4B i32 count>
<4B i32 deleted>
<1B u8  fmt>
<count B alive mask>          ; 1 = alive, 0 = deleted
<count × dim × elem_size B>   ; vector data, f32 or f16
<4B u32 crc32>                ; over alive mask + vector data
```

### `.meta` — unchanged from 1.x
```
<4B i32 count>
for slot 0..count-1:
  <4B i32 lbl_len>            ; 0 = no label
  [lbl_len bytes]
```

### `.data` — **NEW**
```
<4B i32 count>
<count B alive mask>          ; bit i = 1 if slot i has data, 0 = empty
                              ; (separate from .tensors alive mask;
                              ;  vector-deleted slots also have data alive=0)
for each slot with mask bit set, in slot order:
  <4B u32 data_len>           ; ≤ 102400
  [data_len bytes]
<4B u32 crc32>                ; over alive mask + all packed records
```

Empty DB cost: `4 + count + 4` bytes (header + mask + crc32). A DB with zero blobs stores ~`count/8` bytes of payload (it's a byte mask, not bits — matches `.tensors` style, room for cleanup later).

> Decision: `.data` mask is one byte per slot (matches `.tensors` for code-reuse symmetry), even though one bit would suffice. Keeps load/save loops trivial.

### `.hashes` — **NEW**
```
<4B i32 count>                ; equals .tensors count
<count × 8B u64 hash>         ; xxh64 of stored vector bytes per slot;
                              ; 0 for tombstoned slots (entry not in index)
<4B u32 crc32>                ; over the hash array
```

Auxiliary file. On missing or CRC mismatch, the hash index is rebuilt at startup by scanning `.tensors` (xxh64 over each alive slot's stored bytes). Rebuild is O(N) but fast — no behavioral change, only a warmup cost.

### Save order

Sequential, in this order, in place (no temp+rename — matches 1.x):
1. `.tensors`
2. `.meta`
3. `.hashes`
4. `.data`

Load order at startup: same. CRC mismatch on `.tensors`/`.meta`/`.data` → warn + load what's intact, mark `g_crc_ok = 0`. CRC mismatch or missing on `.hashes` → rebuild from `.tensors`.

---

## 6. In-memory state additions

```
unsigned char **g_blobs;     // NULL = no blob
uint32_t      *g_blob_lens;  // 0 if g_blobs[i] == NULL
int            g_blobs_cap;  // grows alongside g_labels_cap
```

Lifecycle hooks (must mirror `g_labels` exactly):
- `gpu_realloc_if_needed`: extend, zero new slots
- `vec_free`: free each non-NULL `g_blobs[i]`, then arrays
- `vec_set_blob(slot, bytes, len)`: free old, malloc new (or NULL if len=0)
- DELETE: `vec_set_blob(idx, NULL, 0)`
- UNDO: `vec_set_blob(g_count, NULL, 0)` after decrement

---

## 7. Error envelope examples

A request that fails parsing or validation returns:
```
01 0E 00 00 00  "err bad index"
```
- `0x01` status
- `0x0000000E` body_len (14)
- 14 ASCII bytes

Errors are ASCII text for human readability; no error code enum (matches 1.x messages, simplifies migration).

---

## 8. Locked decisions

1. **PUSH label source**: header `label_len` / `label_bytes`. Body carries only vector and optional data.
2. **PUSH data requires label**: a record with a payload must have a name; `data_len > 0` with `label_len == 0` is rejected.
3. **UNDO scope**: removes only the last PUSH (not arbitrary DELETEs). Matches 1.x.
4. **`.data` alive mask**: byte-per-slot (matches `.tensors` for code-reuse symmetry).
5. **INFO response**: includes 1B protocol version byte at end of body (always `0x02` in this version).

---

## 9. Compatibility

- **No 1.x compatibility.** This is a clean break. Old SDKs cannot speak 2.0; new SDKs cannot speak 1.x.
- File formats: `.tensors` and `.meta` are unchanged, so existing DBs load. New `.data` file is optional — its absence means "no blobs" (server creates on first SAVE).
- Recommended migration: start a 2.0 server on an existing DB, optionally PUSH SET_DATA to enrich records, SAVE.

---

## 10. Summary command table

| Code | Name | New in 2.0 | Notes |
|---|---|---|---|
| `0x01` | PUSH | modified | label & data optional, in body |
| `0x02` | QUERY | modified | merges PULL+CPULL, metric+shape bytes |
| `0x03` | — | removed | was CPULL |
| `0x04` | GET | modified | merges GET+MGET, mode+shape bytes |
| `0x05` | — | removed | was MGET |
| `0x06` | UPDATE | unchanged | |
| `0x07` | DELETE | modified | also clears label + data |
| `0x08` | CMD_LABEL | unchanged | |
| `0x09` | UNDO | modified | also clears last label + data |
| `0x0A` | SAVE | response now binary | |
| `0x0D` | CLUSTER | response now binary | |
| `0x0E` | DISTINCT | legacy text response | |
| `0x0F` | REPRESENT | legacy text response | |
| `0x10` | INFO | response adds protocol_version byte | |
| `0x11` | QID | modified | merges PID+CPID, metric+shape bytes |
| `0x12` | — | removed | was CPID |
| `0x13` | SET_DATA | **new** | |
| `0x14` | GET_DATA | **new** | |
| `0x15` | EXISTS | **new** | exact-match lookup by vector |
