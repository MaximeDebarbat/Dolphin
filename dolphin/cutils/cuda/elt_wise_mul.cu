
#include <stdint.h>

__global__ void elt_wise_mul_{{ dtype }}({{ dtype }} *x,
                                   {{ dtype }} *y,
                                   {{ dtype }} *z,
                                   uint32_t n){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        z[i] = x[i] * y[i];
    }
}
