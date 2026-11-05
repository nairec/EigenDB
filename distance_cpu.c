#include "vectorial_engine.h"

void vect_compute_dist_cpu(const float* query, const float* dataset, float* out_distances, size_t n, size_t dim, VectMetric metric)
{
    const float* v; // pointer to compared vector from dataset

    if (metric == VECT_METRIC_COSINE) // cosine
    {
        for (int i = 0; i < n; i++)
        {
            v = &dataset[i * dim];
            float dot_product = 0;
            for (int j = 0; j < dim; j++)
            {
                dot_product += query[j] * v[j];
            }
            out_distances[i] = 1 - dot_product;
        }
    } else // L2 squared
    {
        for (int i = 0; i < n; i++)
        {
            v = &dataset[i * dim];
            float sum = 0;
            for (int j = 0; j < dim; j++)
            {
                float d = query[j] - v[j];
                sum += d * d;
            }
            out_distances[i] = sum;
        }
    }
}