
#include <stdint.h>

__global__ void scal_div_{{ dtype }}({{ dtype }} *x,
                                   {{ dtype }} *z,
                                   {{ dtype }} a,
                                   uint32_t n){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        z[i] = x[i] / a;
    }
}
