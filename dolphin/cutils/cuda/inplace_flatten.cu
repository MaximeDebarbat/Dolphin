
#include <stdint.h>
#ifnef MAX_DIM
#define MAX_DIM 32
#endif // MAX_DIM

__global__ void discontiguous_copy_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint32_t* __restrict__ shape,
    const uint32_t* __restrict__ strides,
    const uint32_t ndim,
    const uint32_t size,
){

    uint32_t indexes[MAX_DIM][2];

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size_dst;
         i += blockDim.x * gridDim.x){

        for(uint32_t j = 0; j<ndim; j++){
            indexes[j][0] = index_transform(i, strides_src, shape_src, ndim_src);
            indexes[j][1] = src[indexes[j][0]];
        }

        for(uint32_t j = 0; j<ndim; j++){
            indexes[j][0] = index_transform(i, strides_src, shape_src, ndim_src);
            indexes[j][1] = src[indexes[j][0]];
        }

    }

}