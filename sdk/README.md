# VEC 2.0 SDK

Binary client libraries for `vec` (GPU) and `vec-cpu` (CPU). VEC 2.0 is a clean break from 1.x — see `MIGRATION.md` for the upgrade guide and `PROTOCOL-2.0.md` for the byte-exact spec.

## Transport

| Method | Address |
|--------|---------|
| TCP | `host:port` (default 1920) |
| Named pipe | `\\.\pipe\vec_<name>` (Windows) |
| Unix socket | `/tmp/vec_<name>.sock` (Linux) |

## Frame envelope

### Request
```
F0 <2B ns_len> [ns] <CMD> <2B label_len> [label] <4B body_len> [body]
```

| Field | Size | |
|-------|------|-|
| `F0` | 1 | Magic |
| ns_len | 2 LE | Namespace length. 0 = direct |
| ns | ns_len | Namespace bytes (router mode) |
| CMD | 1 | Command byte |
| label_len | 2 LE | Label length, ≤ 2048 |
| label | label_len | Label bytes (UTF-8) |
| body_len | 4 LE | Body length |
| body | body_len | Command-specific payload |

### Response
```
<1B status> <4B body_len> [body]
```

| Field | Size | |
|-------|------|-|
| status | 1 | `0x00` = OK, `0x01` = ERR |
| body_len | 4 LE | Body length |
| body | body_len | Command-specific (OK) or ASCII error text (ERR) |

Little-endian throughout. Vectors are raw fp32 on the wire even when the DB is f16 (server converts).

## Shape mask

Used by **QUERY**, **QID**, **GET** to control what each result record carries:

| Bit | Meaning |
|---|---|
| `0x01` | include vector |
| `0x02` | include label |
| `0x04` | include data |

Default `0x07` (full record) is what callers want when replacing a MySQL+vec stack. Lean shapes (e.g. `0x02` = label only) are opt-in optimizations.

A result record under shape mask is:
```
<4B i32 index>
<4B f32 distance>             ; only for QUERY/QID
[<4B u32 lbl_len>[lbl_bytes]] ; if shape & 0x02
[<4B u32 dat_len>[dat_bytes]] ; if shape & 0x04
[<dim×4B vector>]             ; if shape & 0x01
```

A response body is `<4B u32 count>` followed by `count` records.

## Commands

| CMD | Hex | Body | Response body |
|-----|-----|------|---------------|
| PUSH | `01` | `vec` + `<4B u32 dlen>[data]` (data requires label) | `<4B i32 index>` |
| QUERY | `02` | `<1B metric><1B shape><vec>` | top-K records (with distance) |
| GET | `04` | `<1B mode><1B shape>[<4B i32 idx>` or `<4B count><count×i32>` or empty if label]` | records (no distance) |
| UPDATE | `06` | `[<4B idx>]<vec>` | empty |
| DELETE | `07` | `<4B idx>` or empty (label) | empty |
| LABEL | `08` | `<4B idx>` (label in header) | empty |
| UNDO | `09` | empty | empty |
| SAVE | `0A` | empty | `<4B u32 saved><4B u32 crc>` |
| CLUSTER | `0D` | `<4B f32 eps><1B mode><4B i32 min_pts>` | legacy text body (lines + `end\n`) |
| DISTINCT | `0E` | `<4B i32 k><1B mode>` | legacy text body |
| REPRESENT | `0F` | `<4B f32 eps><1B mode><4B i32 min_pts>` | legacy text body |
| INFO | `10` | empty | see below |
| QID | `11` | `<1B metric><1B shape>[<4B idx>` or empty if label]` | top-K records (with distance) |
| SET_DATA | `13` | `[<4B idx>]<4B u32 dlen>[data]` | empty |
| GET_DATA | `14` | `<4B idx>` or empty (label) | `<4B u32 dlen>[data]` |

Removed: `0x03` (CPULL → use QUERY metric=1), `0x05` (MGET → use GET batch mode), `0x12` (CPID → use QID metric=1).
Reserved: `0x0B`, `0x0C` (do not reuse).

### Field semantics

- **metric**: `0x00` = L2 (default), `0x01` = cosine. Other values → ERR.
- **mode** (GET): `0x00` = single, `0x01` = batch.
- **PUSH**: vector required; label optional via header; data optional in body but **requires label**. Caps: label ≤ 2048 bytes, data ≤ 102400 bytes.
- **GET single by label** may return multiple records if the label is ambiguous.
- **DELETE / UNDO** clear the label and data of the affected slot in addition to the vector.
- **UPDATE** overwrites the vector only; use `LABEL` and `SET_DATA` for label/data changes.

### INFO response

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 LE | dim |
| 4 | 4 LE | count |
| 8 | 4 LE | deleted |
| 12 | 1 | format (0=f32, 1=f16) |
| 13 | 8 LE | mtime (unix epoch) |
| 21 | 4 LE | CRC32 |
| 25 | 1 | CRC status (0=bad, 1=ok, 2=unknown) |
| 26 | 4 LE | name_len |
| 30 | name_len | database name |
| 30+name_len | 1 | protocol version (always `0x02`) |

## Errors

ERR responses carry an ASCII message in the body. Common ones:

| Error | Cause |
|-------|-------|
| `body too short` / `body length mismatch` | Frame body doesn't match expected layout |
| `bad query body` / `bad get body` / `bad qid body` | Mode/shape/payload mismatch |
| `bad metric` | metric byte not 0 or 1 |
| `bad mode` | GET mode byte not 0 or 1 |
| `data requires label` | PUSH with data but no label |
| `data too large` | data > 102400 bytes |
| `label too long` | label > 2048 bytes |
| `label has invalid chars` | label fails filename validation |
| `label empty` | label is whitespace/empty after sanitization |
| `index out of range` | slot doesn't exist |
| `ambiguous label` | label resolves to multiple slots |
| `label not found` | no slot with this label |
| `deleted` / `already deleted` | tombstoned slot |
| `read-only mode` | write command on read-only DB |
| `unknown binary command` | unrecognized CMD byte |

Read-only rejects: PUSH, UPDATE, DELETE, LABEL, UNDO, SAVE, SET_DATA.

## Libraries

| File | Language | Status |
|------|----------|--------|
| `vec_client.h` | C++ (header-only) | 2.0 |
| `vec_client.py` | Python 3 (numpy) | 2.0 |
| `vec_client.js` | Node.js | 2.0 |
| `vec_client.pas` | Delphi | 2.0 |

All clients expose:
- `push(vec, label=None, data=None)` — vector + optional label + optional data
- `query(vec, cosine=False, shape=FULL)` / `qid(idx_or_label, ...)` — kNN search
- `get(target, shape=FULL)` — single by index, single by label (may yield multiple), or batch
- `set_data(idx_or_label, bytes)` / `get_data(idx_or_label)`
- `update(idx_or_label, vec)` — vector only
- `set_label(idx, label)` / `delete(idx_or_label)` / `undo()`
- `save() → (count, crc)` / `info() → metadata`
- `cluster(eps)` / `distinct(k)` / `represent(eps)` (DISTINCT/REPRESENT are CPU-build stubs)

## Router

`vec --route` or `vec deploy` mode. Set namespace on the client; SDK encodes it in `ns_len`/`ns` fields. Router strips the namespace and forwards to the matching backend pipe/socket. The 2.0 `body_len` is preserved across forwarding.

## File format

```
.tensors  [4B dim][4B count][4B deleted][1B fmt][count×1B alive][vectors][4B CRC32]
.meta     [4B count][per slot: 4B len + label bytes]
.data     [4B count][count×1B alive mask][per present slot: 4B u32 dlen + data bytes][4B CRC32]
```

`.tensors` and `.meta` are unchanged from 1.x — existing DBs load. `.data` is new in 2.0; absent file means "no blobs". Server creates `.data` on first SAVE if any slot has data.
