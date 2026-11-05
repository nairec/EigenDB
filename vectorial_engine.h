#ifndef VECTORIAL_ENGINE_H
#define VECTORIAL_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VectMetric {
    VECT_METRIC_L2_SQUARED = 0, /* sum_i (q_i - x_i)^2 */
    VECT_METRIC_COSINE     = 1  /* 1 - dot(q, x) */
} VectMetric;

typedef struct VectEngine VectEngine;

typedef struct VectSearchResult {
    uint64_t id;
    float    distance;
} VectSearchResult;

/*
    Computes de distance from a query vector to every other vector
    @param dataset: row-major, n * dim floats.
    @param query: dim floats. out_distances: n floats.
    @param Punteros de host. Alineación de 32 bytes para AVX2. 
    If query vector is in dataset, its distance will return 0. Exclude it from the dataset if this is not wanted 
    If metric is different from 0 and 1, it will default to L2 squared */
void vect_compute_dist_cpu(
    const float *query,
    const float *dataset,
    float *out_distances,
    size_t n,
    size_t dim,
    VectMetric metric);

// Pending of implementation
void vect_compute_dist_cuda(
    const float *d_query,
    const float *d_dataset,
    float *d_out_distances,
    size_t n,
    size_t dim,
    VectMetric metric);

#ifdef __cplusplus
}
#endif

#endif /* VECTORIAL_ENGINE_H */