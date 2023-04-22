
#include <stdint.h>
#include <assert.h>

__global__ void elt_wise_div_{{ dtype }}({{ dtype }} *x,
                                   {{ dtype }} *y,
                                   {{ dtype }} *z,
                                   uint32_t n,
                                   uint8_t *error){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        if(y[i] == 0){
            z[i] = ({{dtype}}) 0;
            *error = 1;
            return;
        }

        z[i] = x[i] / y[i];
    }
}
