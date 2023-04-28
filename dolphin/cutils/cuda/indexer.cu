
#include <stdint.h>

__global__ void indexer_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint32_t* __restrict__ shape,
    const uint32_t* __restrict__ strides,
    const uint32_t ndim,
    const uint32_t size
){

    for(uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size; tid += blockDim.x * gridDim.x){

        uint32_t idx = tid;
        uint32_t new_index = 0;

        for (uint32_t i = 0; i < ndim; i++){
            uint32_t r_i = ndim - i - 1;
            new_index += (idx % shape[r_i]) * strides[r_i];
            idx = (uint32_t) idx / shape[r_i];
        }
        dst[tid] = src[new_index];

    }
}