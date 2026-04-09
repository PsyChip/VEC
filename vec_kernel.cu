/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * vec - dead simple GPU-resident vector database (CUDA kernels, fp32)
 */
#include <cstddef>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  L2 distance: fp32 database, fp32 query, fp32 output               */
/* ------------------------------------------------------------------ */

__global__ void kernel_l2_dist(const float* db, const float* query,
                               float* dists, int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* v = db + (size_t)i * dim;
    float sum = 0.0f;

    for (int d = 0; d < dim; d++) {
        float diff = v[d] - query[d];
        sum += diff * diff;
    }

    dists[i] = sum;
}

extern "C"
void launch_l2_dist(const float* db, const float* query,
                    float* dists, int n, int dim)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    kernel_l2_dist<<<blocks, threads>>>(db, query, dists, n, dim);
}

/* ------------------------------------------------------------------ */
/*  Cosine distance: fp32 database, fp32 query, fp32 output            */
/*  cosine_dist = 1 - (a·b) / (||a|| * ||b||)                         */
/* ------------------------------------------------------------------ */

__global__ void kernel_cos_dist(const float* db, const float* query,
                                float* dists, int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* v = db + (size_t)i * dim;
    float dot = 0.0f;
    float norm_v = 0.0f;
    float norm_q = 0.0f;

    for (int d = 0; d < dim; d++) {
        float vf = v[d];
        float qf = query[d];
        dot    += vf * qf;
        norm_v += vf * vf;
        norm_q += qf * qf;
    }

    float denom = sqrtf(norm_v) * sqrtf(norm_q);
    dists[i] = (denom > 0.0f) ? (1.0f - dot / denom) : 1.0f;
}

extern "C"
void launch_cos_dist(const float* db, const float* query,
                     float* dists, int n, int dim)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    kernel_cos_dist<<<blocks, threads>>>(db, query, dists, n, dim);
}
