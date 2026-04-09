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
