#include <stdint.h>

__global__ void invscal_div_{{ dtype }}({{ dtype }} *x,
                                   {{ dtype }} *z,
                                   {{ dtype }} a,
                                   uint32_t n,
                                   uint8_t *error){

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        if(x[i] == 0){
            z[i] = ({{dtype}}) 0;
            *error = 1;
            return;
        }
        z[i] = a / x[i];
    }
}
