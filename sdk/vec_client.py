"""
VEC 2.0 Python Client SDK (binary frame protocol)

VEC 2.0 is a clean break from 1.x. SDKs from 1.x are not wire-compatible.
See PROTOCOL-2.0.md for the full spec.

Usage:
    from vec_client import VecClient, SHAPE_VECTOR, SHAPE_LABEL, SHAPE_DATA, SHAPE_FULL

    vec = VecClient("localhost", 1920)

    # Push: vector required; data requires a label
    idx = vec.push([0.1, 0.2, 0.3])
    idx = vec.push([0.1, 0.2, 0.3], label="docs/file.pdf")
    idx = vec.push([0.1, 0.2, 0.3], label="img/cat.jpg", data=open("cat.jpg","rb").read())

    # Query: returns list of VecResult with optional label/data/vector per shape mask
    results = vec.query([0.1, 0.2, 0.3])                       # default: full record
    results = vec.query([0.1, 0.2, 0.3], cosine=True)
    results = vec.query([0.1, 0.2, 0.3], shape=SHAPE_LABEL)    # lean: index+label only

    # QID: query using a stored vector by index or label
    results = vec.qid(42)
    results = vec.qid("img/cat.jpg", cosine=True)

    # GET: single or batch fetch
    rec  = vec.get(42)                          # single, full record
    recs = vec.get([0, 1, 2])                   # batch
    recs = vec.get("img/cat.jpg")               # by label (may return multiple)

    # Sidecar payload management
    vec.set_data(42, b"...binary blob...")      # ≤100KB
    blob = vec.get_data(42)

    vec.update(42, [0.1, 0.2, 0.3])             # vector only; use set_label / set_data for the rest
    vec.set_label(42, "img/cat.jpg")
    vec.delete(42)                              # also clears label and data
    vec.undo()                                  # also clears last label and data

    saved, crc = vec.save()
    info = vec.info()                           # includes 'protocol' (= 2)

    # Clustering still works; CLUSTER body is legacy text inside binary envelope
    clusters, noise = vec.cluster(0.5)

Router mode:
    vec = VecClient("localhost", 1920, namespace="tools")
"""
import socket
import struct
import numpy as np

BIN_MAGIC        = 0xF0
PROTOCOL_VERSION = 0x02

CMD_PUSH      = 0x01
CMD_QUERY     = 0x02
CMD_GET       = 0x04
CMD_UPDATE    = 0x06
CMD_DELETE    = 0x07
CMD_LABEL     = 0x08
CMD_UNDO      = 0x09
CMD_SAVE      = 0x0A
CMD_CLUSTER   = 0x0D
CMD_DISTINCT  = 0x0E
CMD_REPRESENT = 0x0F
CMD_INFO      = 0x10
CMD_QID       = 0x11
CMD_SET_DATA  = 0x13
CMD_GET_DATA  = 0x14

RESP_OK  = 0x00
RESP_ERR = 0x01

SHAPE_VECTOR = 0x01
SHAPE_LABEL  = 0x02
SHAPE_DATA   = 0x04
SHAPE_FULL   = SHAPE_VECTOR | SHAPE_LABEL | SHAPE_DATA  # 0x07 — default

GET_MODE_SINGLE = 0x00
GET_MODE_BATCH  = 0x01

METRIC_L2     = 0x00
METRIC_COSINE = 0x01

MAX_LABEL_BYTES = 2048
MAX_DATA_BYTES  = 102400


class VecError(RuntimeError):
    pass


class VecRecord:
    """One result record from query / qid / get."""
    __slots__ = ("index", "distance", "label", "data", "vector")

    def __init__(self, index, distance=None, label=None, data=None, vector=None):
        self.index    = index
        self.distance = distance
        self.label    = label
        self.data     = data
        self.vector   = vector

    def __repr__(self):
        parts = [f"index={self.index}"]
        if self.distance is not None: parts.append(f"distance={self.distance:.6f}")
        if self.label is not None:    parts.append(f"label={self.label!r}")
        if self.data is not None:     parts.append(f"data=<{len(self.data)} bytes>")
        if self.vector is not None:   parts.append(f"vector=<{len(self.vector)} f32>")
        return "VecRecord(" + ", ".join(parts) + ")"


class VecClient:
    def __init__(self, host="localhost", port=1920, namespace=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self._ns = namespace.encode() if namespace else b""
        self._dim_cache = None

    # ---------- low-level wire helpers ----------

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise VecError("connection closed")
            buf += chunk
        return bytes(buf)

    def _send_frame(self, cmd, label=b"", body=b""):
        if len(label) > MAX_LABEL_BYTES:
            raise VecError(f"label too long ({len(label)} > {MAX_LABEL_BYTES})")
        hdr = (
            bytes([BIN_MAGIC])
            + struct.pack("<H", len(self._ns)) + self._ns
            + bytes([cmd])
            + struct.pack("<H", len(label)) + label
            + struct.pack("<I", len(body))
        )
        self.sock.sendall(hdr + body)

    def _recv_response(self):
        """Read a 2.0 response envelope. Returns body bytes on OK, raises VecError on ERR."""
        hdr = self._recv_exact(5)
        status = hdr[0]
        body_len = struct.unpack_from("<I", hdr, 1)[0]
        body = self._recv_exact(body_len) if body_len > 0 else b""
        if status == RESP_ERR:
            raise VecError(body.decode("utf-8", errors="replace"))
        if status != RESP_OK:
            raise VecError(f"unknown status byte 0x{status:02x}")
        return body

    @staticmethod
    def _label_bytes(label):
        if label is None:
            return b""
        if isinstance(label, str):
            return label.encode()
        return bytes(label)

    def _dim(self):
        if self._dim_cache is None:
            self._dim_cache = self.info()["dim"]
        return self._dim_cache

    # ---------- record decoding (shape-aware) ----------

    @staticmethod
    def _decode_records(body, shape, dim, with_distance):
        if len(body) < 4:
            return []
        count = struct.unpack_from("<I", body, 0)[0]
        off = 4
        out = []
        for _ in range(count):
            (idx,) = struct.unpack_from("<i", body, off); off += 4
            dist = None
            if with_distance:
                (dist,) = struct.unpack_from("<f", body, off); off += 4
            label = None
            data  = None
            vector = None
            if shape & SHAPE_LABEL:
                (ll,) = struct.unpack_from("<I", body, off); off += 4
                if ll > 0:
                    label = body[off:off+ll].decode("utf-8", errors="replace")
                    off += ll
            if shape & SHAPE_DATA:
                (dl,) = struct.unpack_from("<I", body, off); off += 4
                if dl > 0:
                    data = bytes(body[off:off+dl])
                    off += dl
            if shape & SHAPE_VECTOR:
                vector = np.frombuffer(body, dtype=np.float32, count=dim, offset=off).copy()
                off += dim * 4
            out.append(VecRecord(idx, distance=dist, label=label, data=data, vector=vector))
        return out

    # ---------- commands ----------

    def push(self, vector, label=None, data=None):
        """Push a vector. Returns slot index. data requires label."""
        arr = np.asarray(vector, dtype=np.float32)
        lbl = self._label_bytes(label)
        payload = data if data is not None else b""
        if len(payload) > 0 and not lbl:
            raise VecError("data requires label")
        if len(payload) > MAX_DATA_BYTES:
            raise VecError(f"data too large ({len(payload)} > {MAX_DATA_BYTES})")
        body = arr.tobytes() + struct.pack("<I", len(payload)) + payload
        self._send_frame(CMD_PUSH, label=lbl, body=body)
        body = self._recv_response()
        return struct.unpack("<i", body)[0]

    def query(self, vector, cosine=False, shape=SHAPE_FULL):
        """L2 (or cosine) nearest-neighbor search. Returns list of VecRecord."""
        arr = np.asarray(vector, dtype=np.float32)
        body = bytes([METRIC_COSINE if cosine else METRIC_L2, shape & 0xFF]) + arr.tobytes()
        self._send_frame(CMD_QUERY, body=body)
        return self._decode_records(self._recv_response(), shape, self._dim(), with_distance=True)

    def qid(self, index_or_label, cosine=False, shape=SHAPE_FULL):
        """Query using a stored vector as the source. Accepts int index or str label."""
        metric = METRIC_COSINE if cosine else METRIC_L2
        if isinstance(index_or_label, int):
            body = bytes([metric, shape & 0xFF]) + struct.pack("<i", index_or_label)
            self._send_frame(CMD_QID, body=body)
        else:
            lbl = self._label_bytes(index_or_label)
            body = bytes([metric, shape & 0xFF])
            self._send_frame(CMD_QID, label=lbl, body=body)
        return self._decode_records(self._recv_response(), shape, self._dim(), with_distance=True)

    def get(self, target, shape=SHAPE_FULL):
        """Fetch records.

        target: int (single by index) | str (single by label, may return multiple) |
                list[int] (batch by indices).
        Returns list of VecRecord.
        """
        if isinstance(target, list):
            indices = target
            body = (bytes([GET_MODE_BATCH, shape & 0xFF])
                    + struct.pack("<I", len(indices))
                    + struct.pack(f"<{len(indices)}i", *indices))
            self._send_frame(CMD_GET, body=body)
        elif isinstance(target, int):
            body = bytes([GET_MODE_SINGLE, shape & 0xFF]) + struct.pack("<i", target)
            self._send_frame(CMD_GET, body=body)
        else:
            lbl = self._label_bytes(target)
            body = bytes([GET_MODE_SINGLE, shape & 0xFF])
            self._send_frame(CMD_GET, label=lbl, body=body)
        return self._decode_records(self._recv_response(), shape, self._dim(), with_distance=False)

    def update(self, index_or_label, vector):
        """Overwrite vector in-place. Does NOT touch label or data."""
        arr = np.asarray(vector, dtype=np.float32)
        if isinstance(index_or_label, int):
            body = struct.pack("<i", index_or_label) + arr.tobytes()
            self._send_frame(CMD_UPDATE, body=body)
        else:
            self._send_frame(CMD_UPDATE,
                             label=self._label_bytes(index_or_label),
                             body=arr.tobytes())
        self._recv_response()

    def set_label(self, index, label):
        """Set or clear label for a slot. label=None or empty clears."""
        lbl = self._label_bytes(label) if label else b""
        self._send_frame(CMD_LABEL, label=lbl, body=struct.pack("<i", index))
        self._recv_response()

    def delete(self, index_or_label):
        """Tombstone a slot. Also clears its label and data."""
        if isinstance(index_or_label, int):
            self._send_frame(CMD_DELETE, body=struct.pack("<i", index_or_label))
        else:
            self._send_frame(CMD_DELETE, label=self._label_bytes(index_or_label))
        self._recv_response()

    def undo(self):
        """Remove last pushed record. Also clears its label and data."""
        self._send_frame(CMD_UNDO)
        self._recv_response()

    def save(self):
        """Flush to disk. Returns (saved_count, crc32)."""
        self._send_frame(CMD_SAVE)
        body = self._recv_response()
        return struct.unpack("<II", body)

    def info(self):
        """DB metadata: dim, count, deleted, fmt, mtime, crc, crc_ok, name, protocol."""
        self._send_frame(CMD_INFO)
        body = self._recv_response()
        off = 0
        dim, count, deleted = struct.unpack_from("<iii", body, off); off += 12
        fmt    = body[off]; off += 1
        mtime, = struct.unpack_from("<q", body, off); off += 8
        crc,   = struct.unpack_from("<I", body, off); off += 4
        crc_ok = body[off]; off += 1
        name_len, = struct.unpack_from("<I", body, off); off += 4
        name = body[off:off+name_len].decode("utf-8", errors="replace"); off += name_len
        protocol = body[off]; off += 1
        return {
            "dim": dim, "count": count, "deleted": deleted,
            "fmt": "f16" if fmt == 1 else "f32",
            "mtime": mtime, "crc": crc, "crc_ok": crc_ok,
            "name": name, "protocol": protocol,
        }

    def set_data(self, index_or_label, data):
        """Set or clear sidecar payload. data=None or b"" clears."""
        payload = data or b""
        if len(payload) > MAX_DATA_BYTES:
            raise VecError(f"data too large ({len(payload)} > {MAX_DATA_BYTES})")
        if isinstance(index_or_label, int):
            body = struct.pack("<iI", index_or_label, len(payload)) + payload
            self._send_frame(CMD_SET_DATA, body=body)
        else:
            body = struct.pack("<I", len(payload)) + payload
            self._send_frame(CMD_SET_DATA, label=self._label_bytes(index_or_label), body=body)
        self._recv_response()

    def get_data(self, index_or_label):
        """Fetch sidecar payload. Returns bytes (empty if no data)."""
        if isinstance(index_or_label, int):
            self._send_frame(CMD_GET_DATA, body=struct.pack("<i", index_or_label))
        else:
            self._send_frame(CMD_GET_DATA, label=self._label_bytes(index_or_label))
        body = self._recv_response()
        (dlen,) = struct.unpack_from("<I", body, 0)
        return bytes(body[4:4+dlen])

    # ---------- cluster / distinct / represent ----------
    # Body of these responses is still the legacy line-based text from 1.x,
    # wrapped in the 2.0 binary envelope. Decoded here for convenience.

    def cluster(self, eps, cosine=False, min_pts=2):
        """DBSCAN. Returns (clusters, noise). Each is list[list[str]] of label-or-index strings."""
        body = struct.pack("<fBi", float(eps), 1 if cosine else 0, int(min_pts))
        self._send_frame(CMD_CLUSTER, body=body)
        text = self._recv_response().decode("utf-8", errors="replace")
        return self._parse_text_lines_with_end(text)

    def distinct(self, k, cosine=False):
        """Farthest-point sampling. CPU build returns 'not available' error."""
        body = struct.pack("<iB", int(k), 1 if cosine else 0)
        self._send_frame(CMD_DISTINCT, body=body)
        text = self._recv_response().decode("utf-8", errors="replace")
        return self._parse_text_index_lines(text)

    def represent(self, eps, cosine=False, min_pts=2):
        """One representative per cluster. CPU build returns 'not available' error."""
        body = struct.pack("<fBi", float(eps), 1 if cosine else 0, int(min_pts))
        self._send_frame(CMD_REPRESENT, body=body)
        text = self._recv_response().decode("utf-8", errors="replace")
        return self._parse_text_index_lines(text)

    @staticmethod
    def _parse_text_lines_with_end(text):
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line == "end":
                continue
            lines.append([m for m in line.split(",") if m])
        if lines:
            return lines[:-1], lines[-1]
        return [], []

    @staticmethod
    def _parse_text_index_lines(text):
        out = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line == "end":
                continue
            try:
                out.append(int(line))
            except ValueError:
                out.append(line)
        return out

    # ---------- lifecycle ----------

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
