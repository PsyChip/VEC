/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * vec — dead simple GPU-resident vector database (CUDA kernels, fp16)
 */
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* ------------------------------------------------------------------ */
/*  L2 distance: fp16 database, fp16 query, fp32 output               */
/*  Uses float4-style half2 loads for 2x bandwidth efficiency         */
/* ------------------------------------------------------------------ */

__global__ void kernel_l2_dist(const half* db, const half* query,
                               float* dists, int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const half* v = db + (size_t)i * dim;
    float sum = 0.0f;

    /* process pairs via half2 for better throughput */
    int dim2 = dim / 2;
    const half2* v2 = (const half2*)v;
    const half2* q2 = (const half2*)query;

    for (int d = 0; d < dim2; d++) {
        half2 vv = v2[d];
        half2 qq = q2[d];
        half2 diff = __hsub2(vv, qq);
        half2 sq   = __hmul2(diff, diff);
        sum += __half2float(sq.x) + __half2float(sq.y);
    }

    /* handle odd trailing element */
    if (dim & 1) {
        float diff = __half2float(v[dim - 1]) - __half2float(query[dim - 1]);
        sum += diff * diff;
    }

    dists[i] = sum;
}

extern "C"
void launch_l2_dist(const half* db, const half* query,
                    float* dists, int n, int dim)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    kernel_l2_dist<<<blocks, threads>>>(db, query, dists, n, dim);
}

/* ------------------------------------------------------------------ */
/*  Conversion kernel: fp32 -> fp16 on GPU                             */
/* ------------------------------------------------------------------ */

__global__ void kernel_f32_to_f16(const float* src, half* dst, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst[i] = __float2half(src[i]);
}

extern "C"
void launch_f32_to_f16(const float* src, half* dst, int count)
{
    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    kernel_f32_to_f16<<<blocks, threads>>>(src, dst, count);
}

/* ------------------------------------------------------------------ */
/*  Conversion kernel: fp16 -> fp32 on GPU (for save)                  */
/* ------------------------------------------------------------------ */

__global__ void kernel_f16_to_f32(const half* src, float* dst, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst[i] = __half2float(src[i]);
}

extern "C"
void launch_f16_to_f32(const half* src, float* dst, int count)
{
    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    kernel_f16_to_f32<<<blocks, threads>>>(src, dst, count);
}
