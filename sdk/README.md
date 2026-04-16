# VEC SDK

Client libraries and protocol documentation for VEC. SDKs work identically with both `vec` (GPU) and `vec-cpu` (CPU) servers.

## Protocol

TCP text protocol. One command per line (`\n` terminated), one response per line.

### Connection

- **TCP**: connect to `host:port` (default port 1920)
- **Windows named pipe**: `\\.\pipe\vec_<name>`
- **Linux unix socket**: `/tmp/vec_<name>.sock`

### Commands

| Command | Format | Response |
|---------|--------|----------|
| push | `push [label] <floats>\n` | `<slot_index>\n` |
| bpush | `bpush [label]\n<dim*4 bytes>` | `<slot_index>\n` |
| pull | `pull <floats>\n` | `<label:dist,...>\n` |
| cpull | `cpull <floats>\n` | `<label:dist,...>\n` |
| bpull | `bpull\n<dim*4 bytes>` | `<label:dist,...>\n` |
| bcpull | `bcpull\n<dim*4 bytes>` | `<label:dist,...>\n` |
| label | `label <index> <string>\n` | `ok\n` |
| delete | `delete <index>\n` | `ok\n` |
| undo | `undo\n` | `ok\n` |
| save | `save\n` | `ok\n` |
| size | `size\n` | `<count>\n` |

### Push

Send a vector as comma-separated fp32 floats with optional label. Use quotes for labels with spaces.

```
-> push 0.123,0.456,0.789\n
<- 42\n

-> push docs/report.pdf?page=2 0.123,0.456,0.789\n
<- 43\n

-> push "user: hello world" 0.123,0.456,0.789\n
<- 44\n
```

### Binary push

Send a single vector as raw fp32 binary data with optional label.

```
-> bpush\n
-> <dim * 4 bytes little-endian fp32>
<- 42\n

-> bpush "quoted label"\n
-> <dim * 4 bytes little-endian fp32>
<- 43\n
```

### Query (pull / cpull)

`pull` = L2 distance, `cpull` = cosine distance.

```
-> pull 0.123,0.456,0.789\n
<- docs/report.pdf?page=2:0.001234,photos/beach.jpg:0.045100,42:0.234000\n
```

Results: `label:distance` or `index:distance` pairs. Comma-separated, nearest first. Up to 10 results. Parse each result by splitting on the last colon.

### Binary query (bpull / bcpull)

Same as pull/cpull but query vector sent as raw fp32 bytes.

```
-> bpull\n
-> <dim * 4 bytes little-endian fp32>
<- docs/report.pdf?page=2:0.001234,44:0.234000\n
```

`bcpull` uses cosine distance.

### Label

Set or override label for an existing slot. Quotes supported.

```
-> label 42 docs/report.pdf?page=2\n
<- ok\n

-> label 42 "label with spaces"\n
<- ok\n
```

### Errors

```
err dim mismatch: got 3, expected 1024
err index out of range
err already deleted
err empty
err unknown command
```

## File format

### .tensors

```
[4B dim][4B count][4B deleted][1B format][count B alive mask][vector data][4B CRC32]
```

Naming: `<name>_<dim>_<format>.tensors`

### .meta (labels, only created when labels exist)

```
[4B count][per label: 4B length + string bytes]
```

## Client libraries

- `vec_client.h` - C++
- `vec_client.py` - Python (requires numpy)
- `vec_client.js` - Node.js
- `vec_client.pas` - Delphi

All implement:

```
connect(host, port)
push(vector, [label]) -> int
bpush(vector, [label]) -> int
pull(vector) -> results
cpull(vector) -> results
bpull(vector) -> results
bcpull(vector) -> results
setLabel(index, label)
delete(index)
undo()
save()
size() -> int
close()
```

Results contain `index`, `distance`, and `label` fields. When a label is present, `index` is -1 and `label` contains the string. When no label, `index` is the numeric slot ID.

## Router mode

All SDKs support an optional namespace for use with `vec --route`. Set it once after connecting — all commands are automatically prefixed with `command namespace args`.

```python
# Python
vec = VecClient("localhost", 1920, namespace="tools")

# Node.js
const vec = new VecClient('localhost', 1920, 'tools');

# C++
VecClient vec;
vec.connect_tcp("localhost", 1920);
vec.setNamespace("tools");

# Delphi
Vec := TVecClient.Create;
Vec.ConnectTCP('localhost', 1920);
Vec.Namespace := 'tools';
```
