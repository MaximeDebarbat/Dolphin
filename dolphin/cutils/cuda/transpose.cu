
#include <stdint.h>

/*
    Not optimized for speed, just for clarity.
    TODO:
     - Gride-stride loop
     - Shared memory
*/

__global__ void previous_transpose_{{ dtype }}(const {{ dtype }}* __restrict__ src,
                                      {{ dtype }}* __restrict__ dst,
                                      uint32_t *shape,
                                      uint32_t *strides,
                                      uint32_t ndim,
                                      uint32_t size){

    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t copy_index = index;
    if (index >= size) return;

    uint32_t new_index = 0;
    for(uint32_t i = 0; i < ndim; i++){
        uint32_t r_i = ndim - i - 1;
        new_index += (copy_index % shape[r_i]) * strides[r_i];
        copy_index = (uint32_t) copy_index / shape[r_i];
    }

    dst[index] = src[new_index];

}


#include <stdint.h>

__global__ void transpose_{{ dtype }}({{ dtype }} *src,
                                      {{ dtype }} *dst,
                                      uint32_t *shape,
                                      uint32_t *strides,
                                      uint32_t ndim,
                                      uint32_t size){

    for(uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x){
        uint32_t index_copy = index;
        uint32_t new_index = 0;

        for(uint32_t i = 0; i < ndim; i++){
            uint32_t r_i = ndim - i - 1;
            new_index += (index_copy % shape[r_i]) * strides[r_i];
            index_copy = (uint32_t) index_copy / shape[r_i];
        }


        dst[index] = src[new_index];
    }
}