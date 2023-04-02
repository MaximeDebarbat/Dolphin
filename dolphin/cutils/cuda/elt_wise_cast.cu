
#include <stdint.h>

__global__ void {{ indtype }}_to_{{ outdtype }}({{ indtype }} *src, {{ outdtype }} *dst, uint32_t n){

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        //printf("debug : %f = ({{ outdtype }}) %f\n", dst[i], ({{ outdtype }}) src[i]);
        dst[i] = ({{ outdtype }}) src[i];
    }
}
