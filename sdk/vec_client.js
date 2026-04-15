/**
 * VEC Node.js Client SDK
 *
 * Usage:
 *   const VecClient = require('./vec_client');
 *   const vec = new VecClient('localhost', 1920);
 *   await vec.connect();
 *   const idx = await vec.push([0.1, 0.2, 0.3]);
 *   const idx = await vec.push([0.1, 0.2, 0.3], 'docs/file.pdf?page=2');
 *   const results = await vec.pull([0.1, 0.2, 0.3]);
 *   vec.close();
 */
const net = require('net');

class VecClient {
    constructor(host = 'localhost', port = 1920) {
        this.host = host;
        this.port = port;
        this.sock = null;
        this.buffer = '';
        this.pending = [];
    }

    connect() {
        return new Promise((resolve, reject) => {
            this.sock = net.createConnection(this.port, this.host, () => resolve());
            this.sock.on('error', reject);
            this.sock.on('data', (data) => {
                this.buffer += data.toString();
                let nl;
                while ((nl = this.buffer.indexOf('\n')) !== -1) {
                    const line = this.buffer.substring(0, nl);
                    this.buffer = this.buffer.substring(nl + 1);
                    if (this.pending.length > 0) {
                        this.pending.shift()(line);
                    }
                }
            });
        });
    }

    _command(cmd) {
        return new Promise((resolve, reject) => {
            this.pending.push((resp) => {
                if (resp.startsWith('err')) reject(new Error(resp));
                else resolve(resp);
            });
            this.sock.write(cmd);
        });
    }

    _commandBin(header, binData) {
        return new Promise((resolve, reject) => {
            this.pending.push((resp) => {
                if (resp.startsWith('err')) reject(new Error(resp));
                else resolve(resp);
            });
            this.sock.write(header);
            this.sock.write(binData);
        });
    }

    async push(vector, label = null) {
        const csv = vector.map(v => v.toFixed(6)).join(',');
        const cmd = label ? `push ${label} ${csv}\n` : `push ${csv}\n`;
        const resp = await this._command(cmd);
        return parseInt(resp);
    }

    async bpush(vector, label = null) {
        const dim = vector.length;
        const buf = Buffer.alloc(dim * 4);
        for (let i = 0; i < dim; i++) buf.writeFloatLE(vector[i], i * 4);
        const header = label ? `bpush ${label}\n` : 'bpush\n';
        const resp = await this._commandBin(header, buf);
        return parseInt(resp);
    }

    async pull(vector) {
        const csv = vector.map(v => v.toFixed(6)).join(',');
        const resp = await this._command(`pull ${csv}\n`);
        return this._parseResults(resp);
    }

    async cpull(vector) {
        const csv = vector.map(v => v.toFixed(6)).join(',');
        const resp = await this._command(`cpull ${csv}\n`);
        return this._parseResults(resp);
    }

    async bpull(vector) {
        const dim = vector.length;
        const buf = Buffer.alloc(dim * 4);
        for (let i = 0; i < dim; i++) buf.writeFloatLE(vector[i], i * 4);
        const resp = await this._commandBin('bpull\n', buf);
        return this._parseResults(resp);
    }

    async bcpull(vector) {
        const dim = vector.length;
        const buf = Buffer.alloc(dim * 4);
        for (let i = 0; i < dim; i++) buf.writeFloatLE(vector[i], i * 4);
        const resp = await this._commandBin('bcpull\n', buf);
        return this._parseResults(resp);
    }

    async setLabel(index, label) {
        await this._command(`label ${index} ${label}\n`);
    }

    async delete(index) {
        await this._command(`delete ${index}\n`);
    }

    async undo() {
        await this._command('undo\n');
    }

    async save() {
        await this._command('save\n');
    }

    async size() {
        const resp = await this._command('size\n');
        return parseInt(resp);
    }

    close() {
        if (this.sock) this.sock.destroy();
    }

    _parseResults(resp) {
        if (!resp.trim()) return [];
        return resp.trim().split(',').map(pair => {
            const lastColon = pair.lastIndexOf(':');
            if (lastColon < 0) return null;
            const key = pair.substring(0, lastColon);
            const dist = parseFloat(pair.substring(lastColon + 1));
            const idx = parseInt(key);
            if (!isNaN(idx) && idx.toString() === key) {
                return { index: idx, distance: dist, label: null };
            }
            return { index: -1, distance: dist, label: key };
        }).filter(r => r !== null);
    }
}

module.exports = VecClient;
