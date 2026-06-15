/**
 * VEC 2.0 Node.js Client SDK — TypeScript declarations.
 *
 * Local transport only in this version:
 *   - Windows: named pipe  \\.\pipe\vec_<name>
 *   - Linux:   unix socket /tmp/vec_<name>.sock
 *
 * Pair with vec_client.js. See PROTOCOL-2.0.md for the wire spec.
 */

/// <reference types="node" />

export type VecLabel = string | Buffer;
export type VecData  = Buffer | Uint8Array;
export type VecVector = ArrayLike<number>;

/** Bitmask controlling what each result record carries. */
export type Shape = number;

export interface VecRecord {
    index: number;
    /** Present only on query/qid; null for get/exists. */
    distance: number | null;
    /** Present when shape & SHAPE_LABEL; null otherwise or if label was empty. */
    label: string | null;
    /** Present when shape & SHAPE_DATA. */
    data: Buffer | null;
    /** Present when shape & SHAPE_VECTOR. */
    vector: Float32Array | null;
}

export interface VecExistsResult {
    index: number;
    /** Populated only when shape & SHAPE_LABEL. */
    label?: string;
    /** Populated only when shape & SHAPE_DATA. */
    data?: Buffer;
}

export interface VecInfo {
    dim: number;
    count: number;
    deleted: number;
    fmt: 'f16' | 'f32';
    /** Unix epoch seconds. */
    mtime: number;
    crc: number;
    /** 0 = mismatch, 1 = ok, 2 = unknown (no file yet). */
    crcOk: number;
    name: string;
    /** Always 2 in this version. */
    protocol: number;
}

export interface VecSaveResult {
    savedCount: number;
    crc: number;
}

export interface VecClusterEntry {
    /** Slot indices belonging to this cluster. */
    members: number[];
    /**
     * Arithmetic mean of the member vectors (fp32, length = dim of the DB).
     * Use it to spot degenerate-attractor clusters: a centroid with suspiciously
     * small magnitude, or members with high pairwise distance, often holds
     * low-confidence/fallback embeddings rather than a real coherent group.
     */
    centroid: Float32Array;
}

export interface VecClusterResult {
    clusters: VecClusterEntry[];
    /** Indices of slots that didn't fall into any cluster. */
    noise: number[];
}

export interface PushOpts {
    label?: VecLabel | null;
    data?: VecData | null;
}

export interface QueryOpts {
    cosine?: boolean;
    shape?: Shape;
}

export interface ExistsOpts {
    /** SHAPE_VECTOR is ignored; pass SHAPE_LABEL / SHAPE_DATA bits to fetch them. */
    shape?: Shape;
}

export interface GetOpts {
    shape?: Shape;
}

export interface ClusterOpts {
    cosine?: boolean;
    minPts?: number;
}

export interface DistinctOpts {
    cosine?: boolean;
}

export type IndexOrLabel = number | VecLabel;

declare class VecClient {
    /**
     * Connect to the local instance named `name`.
     * Windows -> \\.\pipe\vec_<name>, Linux -> /tmp/vec_<name>.sock.
     */
    constructor(name: string);

    /** Open the underlying pipe / socket. Must be awaited before issuing commands. */
    connect(): Promise<void>;

    /** Insert a record. Vector required; data requires a label. Returns slot index. */
    push(vector: VecVector, opts?: PushOpts): Promise<number>;

    /** Nearest-neighbor search by query vector. */
    query(vector: VecVector, opts?: QueryOpts): Promise<VecRecord[]>;

    /** Byte-exact lookup. Returns null if no match. */
    exists(vector: VecVector, opts?: ExistsOpts): Promise<VecExistsResult | null>;

    /** Nearest-neighbor search using a stored vector (by index or label). */
    qid(indexOrLabel: IndexOrLabel, opts?: QueryOpts): Promise<VecRecord[]>;

    /**
     * Fetch records:
     *   number      -> single by index
     *   number[]    -> batch
     *   string/Buffer -> single by label (may return multiple)
     */
    get(target: number | number[] | VecLabel, opts?: GetOpts): Promise<VecRecord[]>;

    /** Overwrite vector in place. Does not touch label or data. */
    update(indexOrLabel: IndexOrLabel, vector: VecVector): Promise<void>;

    /** Set or clear label for a slot. Empty/null clears. */
    setLabel(index: number, label: VecLabel | null): Promise<void>;

    /** Tombstone a slot. Also clears its label and data. */
    delete(indexOrLabel: IndexOrLabel): Promise<void>;

    /** Remove the last pushed record. Also clears its label and data. */
    undo(): Promise<void>;

    /** Flush .tensors / .meta / .hashes / .data to disk. */
    save(): Promise<VecSaveResult>;

    /** DB metadata snapshot. */
    info(): Promise<VecInfo>;

    /** Set or clear sidecar payload. Empty clears. */
    setData(indexOrLabel: IndexOrLabel, data: VecData | null): Promise<void>;

    /** Fetch sidecar payload. Returns an empty Buffer when the slot has none. */
    getData(indexOrLabel: IndexOrLabel): Promise<Buffer>;

    /** DBSCAN clustering. Returns members + centroid per cluster; centroid is the
     *  arithmetic mean of member vectors (fp32 even for f16 DBs). */
    cluster(eps: number, opts?: ClusterOpts): Promise<VecClusterResult>;

    /** Farthest-point sampling. CPU build returns 'not available' error. */
    distinct(k: number, opts?: DistinctOpts): Promise<Array<number | string>>;

    /** One representative per DBSCAN cluster. CPU build returns 'not available'. */
    represent(eps: number, opts?: ClusterOpts): Promise<Array<number | string>>;

    /** Close the underlying pipe / socket. */
    close(): void;

    // ---- static shape mask constants ----
    static readonly SHAPE_VECTOR: Shape;
    static readonly SHAPE_LABEL:  Shape;
    static readonly SHAPE_DATA:   Shape;
    /** SHAPE_VECTOR | SHAPE_LABEL | SHAPE_DATA — the default for query/qid/get. */
    static readonly SHAPE_FULL:   Shape;
    static readonly PROTOCOL_VERSION: 2;
}

export = VecClient;
