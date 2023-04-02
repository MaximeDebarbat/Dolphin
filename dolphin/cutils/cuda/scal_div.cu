#include <stdint.h>

__global__ void scal_div_{{ dtype }}({{ dtype }} *x,
                                   {{ dtype }} *z,
                                   {{ dtype }} a,
                                   uint32_t n){

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        z[i] = x[i] / a;
    }
}
