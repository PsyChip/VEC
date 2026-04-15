# VEC SDK

Client libraries and protocol documentation for VEC - GPU-resident vector database.

## Protocol

VEC uses a text-based TCP protocol. One command per line (`\n` terminated), one response per line.

### Connection

- **TCP**: connect to `host:port` (default port 1920)
- **Windows named pipe**: `\\.\pipe\vec_<name>`
- **Linux unix socket**: `/tmp/vec_<name>.sock`

### Commands

| Command | Format | Response |
|---------|--------|----------|
| push | `push [label] <floats>\n` | `<slot_index>\n` |
| bpush | `bpush [label]\n<raw bytes>` | `<slot_index>\n` |
| pull | `pull <floats>\n` | `<results>\n` |
| cpull | `cpull <floats>\n` | `<results>\n` |
| bpull | `bpull\n<raw bytes>` | `<results>\n` |
| bcpull | `bcpull\n<raw bytes>` | `<results>\n` |
| label | `label <index> <string>\n` | `ok\n` |
| delete | `delete <index>\n` | `ok\n` |
| undo | `undo\n` | `ok\n` |
| save | `save\n` | `ok\n` |
| size | `size\n` | `<count>\n` |

### push

Send a vector as comma-separated fp32 floats with optional label. Dimension must match server.

```
-> push 0.123,0.456,0.789\n
<- 42\n

-> push docs/report.pdf?page=2 0.123,0.456,0.789\n
<- 43\n
```

### bpush (binary push)

Send a single vector as raw fp32 binary data with optional label. Faster than text push.

```
-> bpush\n
-> <dim * 4 bytes of raw little-endian fp32 data>
<- 42\n

-> bpush docs/report.pdf?page=2\n
-> <dim * 4 bytes of raw little-endian fp32 data>
<- 43\n
```

### pull / cpull (text query)

Query nearest neighbors. `pull` uses L2 distance, `cpull` uses cosine distance.

```
-> pull 0.123,0.456,0.789\n
<- docs/report.pdf?page=2:0.001234,photos/beach.jpg:0.045100,12:0.234000\n
```

Results are `label:distance` or `index:distance` pairs, comma-separated, nearest first. Up to 10 results. Labels must not contain colons or commas.

### bpull / bcpull (binary query)

Same as pull/cpull but query vector is sent as raw fp32 bytes instead of text.

```
-> bpull\n
-> <dim * 4 bytes of raw little-endian fp32 data>
<- docs/report.pdf?page=2:0.001234,photos/beach.jpg:0.045100\n
```

`bcpull` uses cosine distance.

### label

Set or override a label for an existing slot.

```
-> label 42 docs/report.pdf?page=2\n
<- ok\n
```

Labels must not contain colons or commas (they will be stripped with a warning).

### Errors

All errors start with `err`:

```
err dim mismatch: got 3, expected 1024
err index out of range
err already deleted
err empty
err unknown command
```

## File Format

### .tensors (vector data)

Binary file, little-endian:

```
Offset  Size              Description
0       4 bytes (int32)   dimension
4       4 bytes (int32)   total vector count
8       4 bytes (int32)   deleted count
12      1 byte (uint8)    format (0=fp32, 1=fp16)
13      count bytes       alive mask (1=active, 0=deleted)
13+N    count*dim*elem    vector data (elem=4 for fp32, 2 for fp16)
EOF-4   4 bytes (uint32)  CRC32 checksum
```

### .meta (label data)

Companion file, created only when labels exist:

```
Offset  Size              Description
0       4 bytes (int32)   label count
        per label:
          4 bytes (int32) string length (0 = no label)
          N bytes         string data (if length > 0)
```

File naming: `<name>_<dim>_<format>.tensors` + `<name>_<dim>_<format>.meta`

## Client Libraries

- `vec_client.py` - Python
- `vec_client.js` - Node.js
- `vec_client.h` - C++
- `vec_client.pas` - Delphi

All clients implement:
- `connect(host, port)`
- `push(vector, [label]) -> int`
- `bpush(vector, [label]) -> int`
- `pull(vector) -> results`
- `cpull(vector) -> results`
- `bpull(vector) -> results`
- `bcpull(vector) -> results`
- `setLabel(index, label)`
- `delete(index)`
- `undo()`
- `save()`
- `size() -> int`
- `close()`
