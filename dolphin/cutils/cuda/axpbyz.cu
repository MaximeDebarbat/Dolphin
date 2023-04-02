
#include <stdint.h>

__global__ void axpbyz_{{ dtype }}({{ dtype }} *x,
                                   {{ dtype }} *y,
                                   {{ dtype }} *z,
                                   {{ dtype }} a,
                                   {{ dtype }} b,
                                   uint32_t n){

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        z[i] = a*x[i] + b*y[i];
    }
}
