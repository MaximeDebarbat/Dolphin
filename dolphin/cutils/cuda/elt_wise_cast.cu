
#include <stdint.h>

__global__ void {{ indtype }}_to_{{ outdtype }}({{ indtype }} *src, {{ outdtype }} *dst, uint32_t n){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        dst[i] = ({{ outdtype }}) src[i];
    }
}
