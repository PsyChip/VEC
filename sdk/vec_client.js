/**
 * VEC 2.0 Node.js Client SDK (binary frame protocol)
 *
 * VEC 2.0 is a clean break from 1.x. SDKs from 1.x are not wire-compatible.
 * See PROTOCOL-2.0.md for the full spec.
 *
 * Usage:
 *   const VecClient = require('./vec_client');
 *   const vec = new VecClient('localhost', 1920);
 *   await vec.connect();
 *
 *   // Push: vector required; data requires a label
 *   const idx = await vec.push([0.1, 0.2, 0.3]);
 *   const idx = await vec.push([0.1, 0.2, 0.3], { label: 'docs/file.pdf' });
 *   const idx = await vec.push([0.1, 0.2, 0.3], { label: 'img/cat.jpg', data: jpegBuf });
 *
 *   // Query: returns array of records {index,distance,label,data,vector}
 *   const r = await vec.query([0.1, 0.2, 0.3]);                  // default: full
 *   const r = await vec.query([0.1, 0.2, 0.3], { cosine: true });
 *   const r = await vec.query([0.1, 0.2, 0.3], { shape: VecClient.SHAPE_LABEL });
 *
 *   // QID by stored vector
 *   const r = await vec.qid(42);
 *   const r = await vec.qid('img/cat.jpg', { cosine: true });
 *
 *   // GET: single, batch, or by label
 *   const r1 = await vec.get(42);
 *   const r2 = await vec.get([0,1,2]);
 *   const r3 = await vec.get('img/cat.jpg');
 *
 *   await vec.setData(42, jpegBuf);
 *   const blob = await vec.getData(42);
 *
 *   await vec.update(42, [0.1, 0.2, 0.3]);     // vector only
 *   await vec.setLabel(42, 'img/cat.jpg');
 *   await vec.delete(42);                       // also clears label + data
 *   await vec.undo();                           // also clears last label + data
 *
 *   const { savedCount, crc } = await vec.save();
 *   const info = await vec.info();              // .protocol === 2
 *
 *   const { clusters, noise } = await vec.cluster(0.5);
 *
 * Router mode:
 *   const vec = new VecClient('localhost', 1920, 'tools');
 */
const net = require('net');

const BIN_MAGIC        = 0xF0;
const PROTOCOL_VERSION = 0x02;

const CMD_PUSH      = 0x01;
const CMD_QUERY     = 0x02;
const CMD_GET       = 0x04;
const CMD_UPDATE    = 0x06;
const CMD_DELETE    = 0x07;
const CMD_LABEL     = 0x08;
const CMD_UNDO      = 0x09;
const CMD_SAVE      = 0x0A;
const CMD_CLUSTER   = 0x0D;
const CMD_DISTINCT  = 0x0E;
const CMD_REPRESENT = 0x0F;
const CMD_INFO      = 0x10;
const CMD_QID       = 0x11;
const CMD_SET_DATA  = 0x13;
const CMD_GET_DATA  = 0x14;

const RESP_OK  = 0x00;
const RESP_ERR = 0x01;

const SHAPE_VECTOR = 0x01;
const SHAPE_LABEL  = 0x02;
const SHAPE_DATA   = 0x04;
const SHAPE_FULL   = SHAPE_VECTOR | SHAPE_LABEL | SHAPE_DATA;

const GET_MODE_SINGLE = 0x00;
const GET_MODE_BATCH  = 0x01;

const METRIC_L2     = 0x00;
const METRIC_COSINE = 0x01;

const MAX_LABEL_BYTES = 2048;
const MAX_DATA_BYTES  = 102400;

class VecClient {
    constructor(host = 'localhost', port = 1920, namespace = null) {
        this.host = host;
        this.port = port;
        this.sock = null;
        this.buffer = Buffer.alloc(0);
        this.pending = [];
        this._ns = namespace ? Buffer.from(namespace) : Buffer.alloc(0);
        this._dimCache = null;
    }

    connect() {
        return new Promise((resolve, reject) => {
            this.sock = net.createConnection(this.port, this.host, () => resolve());
            this.sock.on('error', reject);
            this.sock.on('data', (data) => {
                this.buffer = Buffer.concat([this.buffer, data]);
                this._drain();
            });
            this.sock.on('close', () => {
                while (this.pending.length > 0) {
                    const req = this.pending.shift();
                    if (req.reject) req.reject(new Error('connection closed'));
                }
            });
        });
    }

    _drain() {
        while (this.pending.length > 0) {
            const req = this.pending[0];
            if (this.buffer.length < req.n) break;
            const chunk = this.buffer.subarray(0, req.n);
            this.buffer = this.buffer.subarray(req.n);
            this.pending.shift();
            req.resolve(Buffer.from(chunk));
        }
    }

    _readBytes(n) {
        return new Promise((resolve, reject) => {
            this.pending.push({ n, resolve, reject });
            this._drain();
        });
    }

    async _recvResponse() {
        const hdr = await this._readBytes(5);
        const status = hdr[0];
        const bodyLen = hdr.readUInt32LE(1);
        const body = bodyLen > 0 ? await this._readBytes(bodyLen) : Buffer.alloc(0);
        if (status === RESP_ERR) {
            throw new Error(body.toString('utf8'));
        }
        if (status !== RESP_OK) {
            throw new Error(`unknown status byte 0x${status.toString(16)}`);
        }
        return body;
    }

    _sendFrame(cmd, label = Buffer.alloc(0), body = Buffer.alloc(0)) {
        if (label.length > MAX_LABEL_BYTES) throw new Error(`label too long (${label.length} > ${MAX_LABEL_BYTES})`);
        const head = Buffer.alloc(1 + 2 + this._ns.length + 1 + 2 + label.length + 4);
        let off = 0;
        head[off++] = BIN_MAGIC;
        head.writeUInt16LE(this._ns.length, off); off += 2;
        this._ns.copy(head, off); off += this._ns.length;
        head[off++] = cmd;
        head.writeUInt16LE(label.length, off); off += 2;
        label.copy(head, off); off += label.length;
        head.writeUInt32LE(body.length, off); off += 4;
        this.sock.write(head);
        if (body.length > 0) this.sock.write(body);
    }

    _vecToBuffer(vector) {
        const arr = new Float32Array(vector);
        return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
    }

    _labelBuf(label) {
        if (label === null || label === undefined) return Buffer.alloc(0);
        return Buffer.isBuffer(label) ? label : Buffer.from(String(label));
    }

    async _dim() {
        if (this._dimCache === null) this._dimCache = (await this.info()).dim;
        return this._dimCache;
    }

    _decodeRecords(body, shape, dim, withDistance) {
        if (body.length < 4) return [];
        const count = body.readUInt32LE(0);
        let off = 4;
        const out = [];
        for (let i = 0; i < count; i++) {
            const index = body.readInt32LE(off); off += 4;
            let distance = null;
            if (withDistance) { distance = body.readFloatLE(off); off += 4; }
            let label = null, data = null, vector = null;
            if (shape & SHAPE_LABEL) {
                const ll = body.readUInt32LE(off); off += 4;
                if (ll > 0) { label = body.subarray(off, off + ll).toString('utf8'); off += ll; }
            }
            if (shape & SHAPE_DATA) {
                const dl = body.readUInt32LE(off); off += 4;
                if (dl > 0) { data = Buffer.from(body.subarray(off, off + dl)); off += dl; }
            }
            if (shape & SHAPE_VECTOR) {
                const slice = body.subarray(off, off + dim * 4);
                vector = new Float32Array(new Uint8Array(slice).buffer);
                off += dim * 4;
            }
            out.push({ index, distance, label, data, vector });
        }
        return out;
    }

    // ---------------- commands ----------------

    async push(vector, opts = {}) {
        const { label = null, data = null } = opts;
        const lbl = this._labelBuf(label);
        const payload = data || Buffer.alloc(0);
        const payloadBuf = Buffer.isBuffer(payload) ? payload : Buffer.from(payload);
        if (payloadBuf.length > 0 && lbl.length === 0) throw new Error('data requires label');
        if (payloadBuf.length > MAX_DATA_BYTES) throw new Error(`data too large (${payloadBuf.length} > ${MAX_DATA_BYTES})`);
        const vecBuf = this._vecToBuffer(vector);
        const dlen = Buffer.alloc(4); dlen.writeUInt32LE(payloadBuf.length, 0);
        const body = Buffer.concat([vecBuf, dlen, payloadBuf]);
        this._sendFrame(CMD_PUSH, lbl, body);
        const resp = await this._recvResponse();
        return resp.readInt32LE(0);
    }

    async query(vector, opts = {}) {
        const { cosine = false, shape = SHAPE_FULL } = opts;
        const vecBuf = this._vecToBuffer(vector);
        const head = Buffer.from([cosine ? METRIC_COSINE : METRIC_L2, shape & 0xFF]);
        this._sendFrame(CMD_QUERY, Buffer.alloc(0), Buffer.concat([head, vecBuf]));
        const body = await this._recvResponse();
        return this._decodeRecords(body, shape, await this._dim(), true);
    }

    async qid(indexOrLabel, opts = {}) {
        const { cosine = false, shape = SHAPE_FULL } = opts;
        const head = Buffer.from([cosine ? METRIC_COSINE : METRIC_L2, shape & 0xFF]);
        if (typeof indexOrLabel === 'number') {
            const idxBuf = Buffer.alloc(4); idxBuf.writeInt32LE(indexOrLabel, 0);
            this._sendFrame(CMD_QID, Buffer.alloc(0), Buffer.concat([head, idxBuf]));
        } else {
            this._sendFrame(CMD_QID, this._labelBuf(indexOrLabel), head);
        }
        const body = await this._recvResponse();
        return this._decodeRecords(body, shape, await this._dim(), true);
    }

    async get(target, opts = {}) {
        const { shape = SHAPE_FULL } = opts;
        if (Array.isArray(target)) {
            const head = Buffer.from([GET_MODE_BATCH, shape & 0xFF]);
            const cnt = Buffer.alloc(4); cnt.writeUInt32LE(target.length, 0);
            const idxBuf = Buffer.alloc(target.length * 4);
            for (let i = 0; i < target.length; i++) idxBuf.writeInt32LE(target[i], i * 4);
            this._sendFrame(CMD_GET, Buffer.alloc(0), Buffer.concat([head, cnt, idxBuf]));
        } else if (typeof target === 'number') {
            const head = Buffer.from([GET_MODE_SINGLE, shape & 0xFF]);
            const idxBuf = Buffer.alloc(4); idxBuf.writeInt32LE(target, 0);
            this._sendFrame(CMD_GET, Buffer.alloc(0), Buffer.concat([head, idxBuf]));
        } else {
            const head = Buffer.from([GET_MODE_SINGLE, shape & 0xFF]);
            this._sendFrame(CMD_GET, this._labelBuf(target), head);
        }
        const body = await this._recvResponse();
        return this._decodeRecords(body, shape, await this._dim(), false);
    }

    async update(indexOrLabel, vector) {
        const vecBuf = this._vecToBuffer(vector);
        if (typeof indexOrLabel === 'number') {
            const idxBuf = Buffer.alloc(4); idxBuf.writeInt32LE(indexOrLabel, 0);
            this._sendFrame(CMD_UPDATE, Buffer.alloc(0), Buffer.concat([idxBuf, vecBuf]));
        } else {
            this._sendFrame(CMD_UPDATE, this._labelBuf(indexOrLabel), vecBuf);
        }
        await this._recvResponse();
    }

    async setLabel(index, label) {
        const lbl = label ? this._labelBuf(label) : Buffer.alloc(0);
        const idxBuf = Buffer.alloc(4); idxBuf.writeInt32LE(index, 0);
        this._sendFrame(CMD_LABEL, lbl, idxBuf);
        await this._recvResponse();
    }

    async delete(indexOrLabel) {
        if (typeof indexOrLabel === 'number') {
            const idxBuf = Buffer.alloc(4); idxBuf.writeInt32LE(indexOrLabel, 0);
            this._sendFrame(CMD_DELETE, Buffer.alloc(0), idxBuf);
        } else {
            this._sendFrame(CMD_DELETE, this._labelBuf(indexOrLabel));
        }
        await this._recvResponse();
    }

    async undo() {
        this._sendFrame(CMD_UNDO);
        await this._recvResponse();
    }

    async save() {
        this._sendFrame(CMD_SAVE);
        const body = await this._recvResponse();
        return { savedCount: body.readUInt32LE(0), crc: body.readUInt32LE(4) };
    }

    async info() {
        this._sendFrame(CMD_INFO);
        const body = await this._recvResponse();
        let off = 0;
        const dim     = body.readInt32LE(off); off += 4;
        const count   = body.readInt32LE(off); off += 4;
        const deleted = body.readInt32LE(off); off += 4;
        const fmt     = body[off] === 1 ? 'f16' : 'f32'; off += 1;
        const mtime   = Number(body.readBigInt64LE(off)); off += 8;
        const crc     = body.readUInt32LE(off); off += 4;
        const crcOk   = body[off]; off += 1;
        const nameLen = body.readUInt32LE(off); off += 4;
        const name    = body.subarray(off, off + nameLen).toString('utf8'); off += nameLen;
        const protocol = body[off]; off += 1;
        return { dim, count, deleted, fmt, mtime, crc, crcOk, name, protocol };
    }

    async setData(indexOrLabel, data) {
        const payload = data ? (Buffer.isBuffer(data) ? data : Buffer.from(data)) : Buffer.alloc(0);
        if (payload.length > MAX_DATA_BYTES) throw new Error(`data too large (${payload.length} > ${MAX_DATA_BYTES})`);
        const dlen = Buffer.alloc(4); dlen.writeUInt32LE(payload.length, 0);
        if (typeof indexOrLabel === 'number') {
            const idxBuf = Buffer.alloc(4); idxBuf.writeInt32LE(indexOrLabel, 0);
            this._sendFrame(CMD_SET_DATA, Buffer.alloc(0), Buffer.concat([idxBuf, dlen, payload]));
        } else {
            this._sendFrame(CMD_SET_DATA, this._labelBuf(indexOrLabel), Buffer.concat([dlen, payload]));
        }
        await this._recvResponse();
    }

    async getData(indexOrLabel) {
        if (typeof indexOrLabel === 'number') {
            const idxBuf = Buffer.alloc(4); idxBuf.writeInt32LE(indexOrLabel, 0);
            this._sendFrame(CMD_GET_DATA, Buffer.alloc(0), idxBuf);
        } else {
            this._sendFrame(CMD_GET_DATA, this._labelBuf(indexOrLabel));
        }
        const body = await this._recvResponse();
        const dlen = body.readUInt32LE(0);
        return Buffer.from(body.subarray(4, 4 + dlen));
    }

    async cluster(eps, opts = {}) {
        const { cosine = false, minPts = 2 } = opts;
        const data = Buffer.alloc(9);
        data.writeFloatLE(eps, 0);
        data[4] = cosine ? METRIC_COSINE : METRIC_L2;
        data.writeInt32LE(minPts, 5);
        this._sendFrame(CMD_CLUSTER, Buffer.alloc(0), data);
        const text = (await this._recvResponse()).toString('utf8');
        const lines = [];
        for (const raw of text.split('\n')) {
            const line = raw.trim();
            if (!line || line === 'end') continue;
            lines.push(line.split(',').filter(m => m));
        }
        const noise = lines.length > 0 ? lines.pop() : [];
        return { clusters: lines, noise };
    }

    async distinct(k, opts = {}) {
        const { cosine = false } = opts;
        const data = Buffer.alloc(5);
        data.writeInt32LE(k, 0);
        data[4] = cosine ? METRIC_COSINE : METRIC_L2;
        this._sendFrame(CMD_DISTINCT, Buffer.alloc(0), data);
        return this._parseIndexLines(await this._recvResponse());
    }

    async represent(eps, opts = {}) {
        const { cosine = false, minPts = 2 } = opts;
        const data = Buffer.alloc(9);
        data.writeFloatLE(eps, 0);
        data[4] = cosine ? METRIC_COSINE : METRIC_L2;
        data.writeInt32LE(minPts, 5);
        this._sendFrame(CMD_REPRESENT, Buffer.alloc(0), data);
        return this._parseIndexLines(await this._recvResponse());
    }

    _parseIndexLines(body) {
        const out = [];
        for (const raw of body.toString('utf8').split('\n')) {
            const line = raw.trim();
            if (!line || line === 'end') continue;
            const n = parseInt(line, 10);
            out.push(!isNaN(n) && n.toString() === line ? n : line);
        }
        return out;
    }

    close() {
        if (this.sock) this.sock.destroy();
    }
}

VecClient.SHAPE_VECTOR = SHAPE_VECTOR;
VecClient.SHAPE_LABEL  = SHAPE_LABEL;
VecClient.SHAPE_DATA   = SHAPE_DATA;
VecClient.SHAPE_FULL   = SHAPE_FULL;
VecClient.PROTOCOL_VERSION = PROTOCOL_VERSION;

module.exports = VecClient;
