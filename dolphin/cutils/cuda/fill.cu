
#include <stdint.h>

__global__ void fill_{{ dtype }}(
    {{ dtype }}* __restrict__ array,
    const {{ dtype }} value,
    const uint32_t* __restrict__ shape,
    const uint32_t* __restrict__ strides,
    const uint32_t ndim,
    const uint32_t size
){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        array[index_transform(i, strides, shape, ndim)] = value;
    }
}

