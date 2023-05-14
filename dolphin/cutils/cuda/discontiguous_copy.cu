
#include <stdint.h>

__global__ void discontiguous_copy_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint32_t* __restrict__ shape_src,
    const uint32_t* __restrict__ strides_src,
    const uint32_t* __restrict__ shape_dst,
    const uint32_t* __restrict__ strides_dst,
    const uint32_t ndim_src,
    const uint32_t ndim_dst,
    const uint32_t size_src,
    const uint32_t size_dst
){

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size_dst;
         i += blockDim.x * gridDim.x){

        uint32_t index_dst = index_transform(i, strides_dst, shape_dst, ndim_dst);
        uint32_t index_src = index_transform(i, strides_src, shape_src, ndim_src);
        dst[index_dst] = src[index_src];
    }

}