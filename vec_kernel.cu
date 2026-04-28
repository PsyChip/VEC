/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * vec - dead simple GPU-resident vector database (CUDA kernels)
 */
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* fp32 L2 */
__global__ void kernel_l2_f32(const float *db, const float *query, float *dists, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *v = db + (size_t)i * dim;
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = v[d] - query[d];
        sum += diff * diff;
    }
    dists[i] = sum;
}

/* fp32 cosine */
__global__ void kernel_cos_f32(const float *db, const float *query, float *dists, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *v = db + (size_t)i * dim;
    float dot = 0.0f, nv = 0.0f, nq = 0.0f;
    for (int d = 0; d < dim; d++) {
        float vf = v[d], qf = query[d];
        dot += vf * qf;
        nv += vf * vf;
        nq += qf * qf;
    }
    float denom = sqrtf(nv) * sqrtf(nq);
    dists[i] = (denom > 0.0f) ? (1.0f - dot / denom) : 1.0f;
}

/* fp16 L2: stored as half, compute in fp32 */
__global__ void kernel_l2_f16(const half *db, const half *query, float *dists, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const half *v = db + (size_t)i * dim;
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = __half2float(v[d]) - __half2float(query[d]);
        sum += diff * diff;
    }
    dists[i] = sum;
}

/* fp16 cosine: stored as half, compute in fp32 */
__global__ void kernel_cos_f16(const half *db, const half *query, float *dists, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const half *v = db + (size_t)i * dim;
    float dot = 0.0f, nv = 0.0f, nq = 0.0f;
    for (int d = 0; d < dim; d++) {
        float vf = __half2float(v[d]), qf = __half2float(query[d]);
        dot += vf * qf;
        nv += vf * vf;
        nq += qf * qf;
    }
    float denom = sqrtf(nv) * sqrtf(nq);
    dists[i] = (denom > 0.0f) ? (1.0f - dot / denom) : 1.0f;
}

/* fp32 -> fp16 conversion kernel */
__global__ void kernel_f32_to_f16(const float *src, half *dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst[i] = __float2half(src[i]);
}

static int calc_blocks(int n) { return (n + 255) / 256; }

extern "C" void launch_l2_f32(const float *db, const float *query, float *dists, int n, int dim) {
    kernel_l2_f32<<<calc_blocks(n), 256>>>(db, query, dists, n, dim);
}

extern "C" void launch_cos_f32(const float *db, const float *query, float *dists, int n, int dim) {
    kernel_cos_f32<<<calc_blocks(n), 256>>>(db, query, dists, n, dim);
}

extern "C" void launch_l2_f16(const void *db, const void *query, float *dists, int n, int dim) {
    kernel_l2_f16<<<calc_blocks(n), 256>>>((const half *)db, (const half *)query, dists, n, dim);
}

extern "C" void launch_cos_f16(const void *db, const void *query, float *dists, int n, int dim) {
    kernel_cos_f16<<<calc_blocks(n), 256>>>((const half *)db, (const half *)query, dists, n, dim);
}

extern "C" void launch_f32_to_f16(const float *src, void *dst, int count) {
    kernel_f32_to_f16<<<calc_blocks(count), 256>>>(src, (half *)dst, count);
}

/* ===================================================================== */
/*  GPU top-K: find K smallest distances                                 */
/* ===================================================================== */

#define TOPK_K 10
#define TOPK_THREADS 32

/* insert (dist, id) into a sorted descending buffer of size K */
__device__ void topk_insert(float *buf_d, int *buf_i, int k, float dist, int id) {
    if (dist >= buf_d[0]) return; /* worse than worst in buffer */
    buf_d[0] = dist;
    buf_i[0] = id;
    /* bubble down — buf[0] is worst (largest), buf[k-1] is best (smallest) */
    for (int j = 0; j < k - 1; j++) {
        if (buf_d[j] <= buf_d[j + 1]) break;
        float td = buf_d[j]; buf_d[j] = buf_d[j + 1]; buf_d[j + 1] = td;
        int ti = buf_i[j]; buf_i[j] = buf_i[j + 1]; buf_i[j + 1] = ti;
    }
}

__global__ void kernel_topk(const float *dists, const unsigned char *alive, int n,
                            float *out_dists, int *out_ids) {
    /* each thread maintains a local top-K buffer */
    float local_d[TOPK_K];
    int local_i[TOPK_K];
    for (int j = 0; j < TOPK_K; j++) { local_d[j] = 3.402823466e+38f; local_i[j] = -1; }

    /* stride through distance array */
    for (int i = threadIdx.x; i < n; i += TOPK_THREADS) {
        if (!alive[i]) continue;
        topk_insert(local_d, local_i, TOPK_K, dists[i], i);
    }

    /* write local results to shared memory for thread 0 to merge */
    __shared__ float s_d[TOPK_THREADS * TOPK_K];
    __shared__ int s_i[TOPK_THREADS * TOPK_K];

    int base = threadIdx.x * TOPK_K;
    for (int j = 0; j < TOPK_K; j++) {
        s_d[base + j] = local_d[j];
        s_i[base + j] = local_i[j];
    }
    __syncthreads();

    /* thread 0 merges all results */
    if (threadIdx.x == 0) {
        float best_d[TOPK_K];
        int best_i[TOPK_K];
        for (int j = 0; j < TOPK_K; j++) { best_d[j] = 3.402823466e+38f; best_i[j] = -1; }

        for (int t = 0; t < TOPK_THREADS; t++) {
            int tb = t * TOPK_K;
            for (int j = 0; j < TOPK_K; j++) {
                topk_insert(best_d, best_i, TOPK_K, s_d[tb + j], s_i[tb + j]);
            }
        }

        /* output sorted ascending: best first */
        for (int j = 0; j < TOPK_K; j++) {
            out_dists[j] = best_d[TOPK_K - 1 - j];
            out_ids[j] = best_i[TOPK_K - 1 - j];
        }
    }
}

extern "C" void launch_topk(const float *d_dists, const unsigned char *d_alive, int n,
                             float *out_dists, int *out_ids) {
    kernel_topk<<<1, TOPK_THREADS>>>(d_dists, d_alive, n, out_dists, out_ids);
}
