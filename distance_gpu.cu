#include <stdio.h>
#include <cuda_runtime.h>
#include "vectorial_engine.h"

enum { VECT_CUDA_BLOCK_SIZE = 256 };

static int vect_cuda_check(cudaError_t err, const char *call)
{
    if (err == cudaSuccess) {
        return 1;
    }
    fprintf(stderr, "CUDA error in %s: %s\n", call, cudaGetErrorString(err));
    return 0;
}

/*
 * One warp computes one dataset vector. 
 * Each warp has 32 lanes and the modular position of a thread is its lane in the warp.
 * (lane of thread = threadIdx.x % 32)
 * Its 32 lanes split the components and lane 0 writes the final distance.
 */
__global__ void vect_dist_kernel(const float *d_query,
                                 const float *d_dataset,
                                 float *d_out_distances,
                                 size_t n,
                                 size_t dim,
                                 int metric)
{
    size_t lane = threadIdx.x % 32; // 32 lanes per warp
    size_t i = (size_t)blockIdx.x * ((size_t)blockDim.x / 32) + (size_t)threadIdx.x / 32; 
    if (i >= n) {
        return;
    }

    const float *row = d_dataset + i * dim; // the row of this warp
    float partial_sum = 0;

    if (metric == VECT_METRIC_COSINE)
    {
        for (size_t j = lane; j < dim; j+= 32) 
        {
            partial_sum += d_query[j] * row[j];
        }
    } else
    {
        for (size_t j = lane; j < dim; j+= 32)
        {
            float d = d_query[j] - row[j];
            partial_sum += d * d;
        }
    
    }

    /*  Fold the 32 partial sums into lane 0
        Each iteration reduces the offset in half and
        sums the partial of (lane_x) + (lane_x + offset)
    */
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_sum += __shfl_down_sync(0xffffffffu, partial_sum, offset); // 0xffffffff marks every lane to sumatory
    }

    // The lane 0 writes the final computed distance on its corresponding position
    if (lane == 0) {
        d_out_distances[i] = (metric == VECT_METRIC_COSINE) ? 1.0f - partial_sum : partial_sum;
    }
}

extern "C" void vect_compute_dist_cuda(const float *d_query,
                                       const float *d_dataset,
                                       float *d_out_distances,
                                       size_t n,
                                       size_t dim,
                                       VectMetric metric)
{
    int device = 0;
    unsigned int grid = 0;

    if (!vect_cuda_check(cudaGetDevice(&device), "cudaGetDevice")) {
        return;
    }
    if (n == 0 || dim == 0) {
        return;
    }

    unsigned int warps_per_block = VECT_CUDA_BLOCK_SIZE / 32;
    grid = (unsigned int)((n + warps_per_block - 1) / warps_per_block);
    vect_dist_kernel<<<grid, VECT_CUDA_BLOCK_SIZE>>>(
        d_query, d_dataset, d_out_distances, n, dim, (int)metric);

    if (!vect_cuda_check(cudaGetLastError(), "vect_dist_kernel launch")) {
        return;
    }
    vect_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}
