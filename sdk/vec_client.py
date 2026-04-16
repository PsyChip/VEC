"""
VEC Python Client SDK

Usage:
    from vec_client import VecClient

    vec = VecClient("localhost", 1920)
    idx = vec.push([0.1, 0.2, 0.3])
    idx = vec.push([0.1, 0.2, 0.3], label="docs/file.pdf?page=2")
    results = vec.pull([0.1, 0.2, 0.3])
    vec.close()

Router mode:
    vec = VecClient("localhost", 1920, namespace="tools")
    idx = vec.push([0.1, 0.2, 0.3])  # sends: push tools 0.1,...
"""
import socket
import struct
import numpy as np

class VecResult:
    def __init__(self, index, distance, label=None):
        self.index = index
        self.distance = distance
        self.label = label

    def __repr__(self):
        if self.label:
            return f"{self.label}:{self.distance:.6f}"
        return f"{self.index}:{self.distance:.6f}"

class VecClient:
    def __init__(self, host="localhost", port=1920, namespace=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self._ns = f"{namespace} " if namespace else ""

    def _send(self, cmd):
        self.sock.sendall(cmd.encode())

    def _send_bytes(self, data):
        self.sock.sendall(data)

    def _readline(self):
        buf = b""
        while True:
            c = self.sock.recv(1)
            if not c or c == b"\n":
                return buf.decode()
            buf += c

    def _check_err(self, resp):
        if resp.startswith("err"):
            raise RuntimeError(resp)
        return resp

    def push(self, vector, label=None):
        """Push a single vector. Returns slot index."""
        csv = ",".join(f"{v:.6f}" for v in vector)
        if label:
            self._send(f"push {self._ns}{label} {csv}\n")
        else:
            self._send(f"push {self._ns}{csv}\n")
        return int(self._check_err(self._readline()))

    def bpush(self, vector, label=None):
        """Binary push - single vector as raw fp32 bytes. Returns slot index."""
        arr = np.array(vector, dtype=np.float32)
        if label:
            self._send(f"bpush {self._ns}{label}\n")
        else:
            self._send(f"bpush {self._ns}\n")
        self._send_bytes(arr.tobytes())
        return int(self._check_err(self._readline()))

    def pull(self, vector):
        """Query by L2 distance. Returns list of VecResult."""
        csv = ",".join(f"{v:.6f}" for v in vector)
        self._send(f"pull {self._ns}{csv}\n")
        return self._parse_results(self._readline())

    def cpull(self, vector):
        """Query by cosine distance. Returns list of VecResult."""
        csv = ",".join(f"{v:.6f}" for v in vector)
        self._send(f"cpull {self._ns}{csv}\n")
        return self._parse_results(self._readline())

    def bpull(self, vector):
        """Binary L2 query - single vector as raw fp32 bytes. Returns list of VecResult."""
        arr = np.array(vector, dtype=np.float32)
        self._send(f"bpull {self._ns}\n")
        self._send_bytes(arr.tobytes())
        return self._parse_results(self._readline())

    def bcpull(self, vector):
        """Binary cosine query - single vector as raw fp32 bytes. Returns list of VecResult."""
        arr = np.array(vector, dtype=np.float32)
        self._send(f"bcpull {self._ns}\n")
        self._send_bytes(arr.tobytes())
        return self._parse_results(self._readline())

    def setLabel(self, index, label):
        """Set or override label for a slot."""
        self._send(f"label {self._ns}{index} {label}\n")
        self._check_err(self._readline())

    def delete(self, index):
        """Tombstone a vector by slot index."""
        self._send(f"delete {self._ns}{index}\n")
        self._check_err(self._readline())

    def undo(self):
        """Remove last pushed vector."""
        self._send(f"undo {self._ns}\n")
        self._check_err(self._readline())

    def save(self):
        """Force save to disk."""
        self._send(f"save {self._ns}\n")
        self._check_err(self._readline())

    def size(self):
        """Return total index count."""
        self._send(f"size {self._ns}\n")
        return int(self._readline())

    def dim(self):
        """Return vector dimension."""
        self._send(f"dim {self._ns}\n")
        return int(self._readline())

    def close(self):
        """Close connection."""
        self.sock.close()

    def _parse_results(self, resp):
        self._check_err(resp)
        if not resp.strip():
            return []
        results = []
        for pair in resp.strip().split(","):
            idx = pair.rfind(":")
            if idx < 0:
                continue
            key = pair[:idx]
            dist = float(pair[idx + 1:])
            try:
                results.append(VecResult(int(key), dist))
            except ValueError:
                results.append(VecResult(-1, dist, label=key))
        return results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
