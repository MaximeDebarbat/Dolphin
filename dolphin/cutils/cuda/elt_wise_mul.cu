
#include <stdint.h>

__global__ void elt_wise_mul_{{ dtype }}({{ dtype }} *x,
                                   {{ dtype }} *y,
                                   {{ dtype }} *z,
                                   uint32_t n){

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        z[i] = x[i] * y[i];
    }
}
