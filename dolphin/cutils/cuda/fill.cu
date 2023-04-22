
#include <stdint.h>

__global__ void fill_{{ dtype }}(
    {{ dtype }}* __restrict__ array,
    const {{ dtype }} value,
    const uint32_t n
){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        array[i] = value;
    }
}

