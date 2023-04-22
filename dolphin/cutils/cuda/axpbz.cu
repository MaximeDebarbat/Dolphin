
#include <stdint.h>

__global__ void axpbz_{{ dtype }}({{ dtype }} *x,
                                  {{ dtype }} *z,
                                  {{ dtype }} a,
                                  {{ dtype }} b,
                                  uint32_t n){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        z[i] = a*x[i] + b;
    }
}
