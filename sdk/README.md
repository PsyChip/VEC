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
| push | `push <comma-separated floats>\n` | `<slot_index>\n` |
| pull | `pull <comma-separated floats>\n` | `<id:dist,id:dist,...>\n` |
| cpull | `cpull <comma-separated floats>\n` | `<id:dist,id:dist,...>\n` |
| bpush | `bpush <count>\n<raw bytes>` | `<first_slot_index>\n` |
| delete | `delete <slot_index>\n` | `ok\n` |
| undo | `undo\n` | `ok\n` |
| save | `save\n` | `ok\n` |
| size | `size\n` | `<count>\n` |

### push

Send a vector as comma-separated fp32 floats. Dimension must match server configuration.

```
-> push 0.123,0.456,0.789\n
<- 42\n
```

Returns the slot index. Store this - it's your key.

### pull (L2 distance)

Query nearest neighbors by Euclidean distance.

```
-> pull 0.123,0.456,0.789\n
<- 0:0.001234,5:0.045100,12:0.234000\n
```

Returns up to 10 results as `index:distance` pairs, comma-separated, sorted nearest first.

### cpull (cosine distance)

Query nearest neighbors by cosine distance.

```
-> cpull 0.123,0.456,0.789\n
<- 0:0.001234,5:0.045100,12:0.234000\n
```

Same format as pull. Use for text embeddings. Use pull for vision embeddings.

### bpush (binary bulk push)

Send multiple vectors as raw fp32 binary data. Much faster than individual text pushes.

```
-> bpush <count>\n
-> <count * dim * 4 bytes of raw little-endian fp32 data>
<- <first_slot_index>\n
```

The server knows the dimension from startup. Expected bytes = `count * dim * sizeof(float)`.

Vectors are laid out contiguously: `[vec0_f0, vec0_f1, ..., vec0_fN, vec1_f0, vec1_f1, ...]`

### Errors

All errors start with `err`:

```
err dim mismatch: got 3, expected 1024
err index out of range
err already deleted
err empty
err unknown command
```

## .tensors File Format

Binary file, little-endian:

```
Offset  Size              Description
0       4 bytes (int32)   dimension
4       4 bytes (int32)   total vector count
8       4 bytes (int32)   deleted count
12      1 byte (uint8)    format (0=fp32, 1=fp16)
13      count bytes       alive mask (1=active, 0=deleted)
13+N    count*dim*elem    vector data (elem=4 for fp32, 2 for fp16)
```

File naming convention: `<name>_<dim>_<format>.tensors`
Example: `mydb_1024_f32.tensors`

## Client Libraries

- `vec_client.py` - Python
- `vec_client.js` - Node.js
- `vec_client.cpp` / `vec_client.h` - C++
- `vec_client.pas` - Delphi

All clients implement the same interface:
- `connect(host, port)` / `connect_pipe(name)`
- `push(vector) -> int`
- `pull(vector) -> [(id, distance)]`
- `cpull(vector) -> [(id, distance)]`
- `bpush(vectors) -> int`
- `delete(index)`
- `undo()`
- `save()`
- `size() -> int`
- `close()`
