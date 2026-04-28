# VEC 2.0 Endpoints

| Code | Name | Purpose | Input body | Output body (OK) | Possible errors |
|---|---|---|---|---|---|
| `0x01` | **PUSH** | Insert record (vector required; label optional via header; data optional, requires label) | `<dim×4B f32 vec><4B u32 dlen>[dlen bytes]` | `<4B i32 index>` | `body too short` · `body length mismatch` · `data too large` · `data requires label` · `label too long` · `label has invalid chars` · `label empty` · `label store failed` · `data store failed` · `read-only mode` |
| `0x02` | **QUERY** | NN search by query vector | `<1B metric><1B shape><dim×4B f32 query>` | `<4B u32 count>` then per record: `<4B i32 index><4B f32 dist>` + optional `[lbl][data][vec]` per shape | `bad query body` · `bad metric` · `out of memory` |
| `0x04` | **GET** | Fetch by index, label, or batch | `<1B mode><1B shape>` then `<4B i32 idx>` (single) / `<4B count><count×i32>` (batch) / empty (label) | `<4B u32 count>` then per record: `<4B i32 index>` + optional `[lbl][data][vec]` per shape | `bad get body` · `bad batch body` · `bad batch length` · `bad mode` · `extra body bytes` · `index out of range` · `deleted` · `label not found` · `out of memory` |
| `0x06` | **UPDATE** | Overwrite vector in place (label/data untouched) | `<dim×4B vec>` (label) or `<4B i32 idx><dim×4B vec>` | empty | `bad body length` · `index out of range` · `deleted` · `ambiguous label` · `label not found` · `read-only mode` |
| `0x07` | **DELETE** | Tombstone slot; also frees label + data | `<4B i32 idx>` or empty (label) | empty | `bad body length` · `extra body bytes` · `index out of range` · `already deleted` · `ambiguous label` · `label not found` · `read-only mode` |
| `0x08` | **CMD_LABEL** | Set or clear label for slot (via header; len=0 clears) | `<4B i32 idx>` | empty | `bad body length` · `index out of range` · `label too long` · `label has invalid chars` · `label empty` · `read-only mode` |
| `0x09` | **UNDO** | Remove last PUSH; also frees its label + data | empty | empty | `extra body bytes` · `empty` · `read-only mode` |
| `0x0A` | **SAVE** | Flush `.tensors` + `.meta` + `.data` to disk | empty | `<4B u32 saved><4B u32 crc32>` | `extra body bytes` · `read-only mode` |
| `0x0D` | **CLUSTER** | DBSCAN clustering | `<4B f32 eps><1B mode><4B i32 min_pts>` | legacy text body: members per line + `end\n` | `bad body length` · `invalid eps` · `bad metric` |
| `0x0E` | **DISTINCT** | Farthest-point sampling | `<4B i32 k><1B mode>` | legacy text body: one index per line + `end\n` | `bad body length` · `invalid k` · `bad metric` · `distinct not available in cpu mode` |
| `0x0F` | **REPRESENT** | One most-distinct member per DBSCAN cluster | `<4B f32 eps><1B mode><4B i32 min_pts>` | legacy text body: one index per line + `end\n` | `bad body length` · `invalid eps` · `bad metric` · `represent not available in cpu mode` |
| `0x10` | **INFO** | DB metadata snapshot | empty | `<4B dim><4B count><4B deleted><1B fmt><8B mtime><4B crc><1B crc_ok><4B name_len>[name]<1B protocol=0x02>` | `extra body bytes` · `out of memory` |
| `0x11` | **QID** | NN search by stored vector / label | `<1B metric><1B shape>` then `<4B i32 idx>` or empty (label) | identical to QUERY | `bad qid body` · `bad metric` · `extra body bytes` · `index out of range` · `deleted` · `ambiguous label` · `label not found` · `out of memory` |
| `0x13` | **SET_DATA** | Set or clear sidecar blob for slot (≤100KB; len=0 clears) | `<4B u32 dlen>[bytes]` (label) or `<4B i32 idx><4B u32 dlen>[bytes]` | empty | `bad body length` · `data too large` · `index out of range` · `deleted` · `ambiguous label` · `label not found` · `data store failed` · `read-only mode` |
| `0x14` | **GET_DATA** | Fetch sidecar blob for slot | `<4B i32 idx>` or empty (label) | `<4B u32 dlen>[bytes]` | `bad body length` · `extra body bytes` · `index out of range` · `deleted` · `ambiguous label` · `label not found` |

**Shape mask:** `0x01` vector · `0x02` label · `0x04` data · default `0x07` (full).
**Metric:** `0x00` L2 (default) · `0x01` cosine.
**Frame envelope:** request `F0 <ns_len><ns><CMD><lbl_len><lbl><body_len><body>` · response `<status><body_len><body>` (status `0x00`=OK, `0x01`=ERR).
**ERR body** is ASCII text; any unrecognized command returns `unknown binary command`.
