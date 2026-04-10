"""
VEC Python Client SDK

Usage:
    from vec_client import VecClient

    vec = VecClient("localhost", 1920)
    idx = vec.push([0.1, 0.2, 0.3])
    results = vec.pull([0.1, 0.2, 0.3])
    vec.close()
"""
import socket
import struct
import numpy as np

class VecClient:
    def __init__(self, host="localhost", port=1920):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def _send(self, cmd):
        self.sock.sendall(cmd.encode())

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

    def push(self, vector):
        """Push a single vector. Returns slot index."""
        csv = ",".join(f"{v:.6f}" for v in vector)
        self._send(f"push {csv}\n")
        return int(self._check_err(self._readline()))

    def pull(self, vector):
        """Query by L2 distance. Returns list of (index, distance) tuples."""
        csv = ",".join(f"{v:.6f}" for v in vector)
        self._send(f"pull {csv}\n")
        return self._parse_results(self._readline())

    def cpull(self, vector):
        """Query by cosine distance. Returns list of (index, distance) tuples."""
        csv = ",".join(f"{v:.6f}" for v in vector)
        self._send(f"cpull {csv}\n")
        return self._parse_results(self._readline())

    def bpush(self, vectors):
        """Binary bulk push. vectors: numpy array or list of lists. Returns first slot index."""
        arr = np.array(vectors, dtype=np.float32)
        count = arr.shape[0]
        self._send(f"bpush {count}\n")
        self.sock.sendall(arr.tobytes())
        return int(self._check_err(self._readline()))

    def delete(self, index):
        """Tombstone a vector by slot index."""
        self._send(f"delete {index}\n")
        self._check_err(self._readline())

    def undo(self):
        """Remove last pushed vector."""
        self._send("undo\n")
        self._check_err(self._readline())

    def save(self):
        """Force save to disk."""
        self._send("save\n")
        self._check_err(self._readline())

    def size(self):
        """Return total index count."""
        self._send("size\n")
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
            idx, dist = pair.split(":")
            results.append((int(idx), float(dist)))
        return results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
