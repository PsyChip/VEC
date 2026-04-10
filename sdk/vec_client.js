/**
 * VEC Node.js Client SDK
 *
 * Usage:
 *   const VecClient = require('./vec_client');
 *   const vec = new VecClient('localhost', 1920);
 *   await vec.connect();
 *   const idx = await vec.push([0.1, 0.2, 0.3]);
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

    async push(vector) {
        const csv = vector.map(v => v.toFixed(6)).join(',');
        const resp = await this._command(`push ${csv}\n`);
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

    async bpush(vectors) {
        const count = vectors.length;
        const dim = vectors[0].length;
        const buf = Buffer.alloc(count * dim * 4);
        let offset = 0;
        for (let i = 0; i < count; i++) {
            for (let d = 0; d < dim; d++) {
                buf.writeFloatLE(vectors[i][d], offset);
                offset += 4;
            }
        }
        const header = `bpush ${count}\n`;
        const resp = await new Promise((resolve, reject) => {
            this.pending.push((resp) => {
                if (resp.startsWith('err')) reject(new Error(resp));
                else resolve(resp);
            });
            this.sock.write(header);
            this.sock.write(buf);
        });
        return parseInt(resp);
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
            const [idx, dist] = pair.split(':');
            return { index: parseInt(idx), distance: parseFloat(dist) };
        });
    }
}

module.exports = VecClient;
