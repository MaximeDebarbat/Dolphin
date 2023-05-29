
#include <stdint.h>

__global__ void {{ indtype }}_to_{{ outdtype }}(const {{ indtype }}* __restrict__ src,
                                                {{ outdtype }}* __restrict__ dst,
                                                const uint32_t* __restrict__ shape_src,
                                                const uint32_t* __restrict__ strides_src,
                                                const uint32_t* __restrict__ shape_dst,
                                                const uint32_t* __restrict__ strides_dst,
                                                const uint32_t ndim,
                                                const uint32_t n){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        uint32_t index_src = index_transform(i, strides_src, shape_src, ndim);
        uint32_t index_dst = index_transform(i, strides_dst, shape_dst, ndim);
        dst[index_dst] = ({{ outdtype }}) src[index_src];
    }
}
