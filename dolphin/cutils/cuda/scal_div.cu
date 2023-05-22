
#include <stdint.h>

__global__ void scal_div_{{ dtype }}(const {{ dtype }}* __restrict__ x,
                                     {{ dtype }}* __restrict__ z,
                                   const uint32_t* __restrict__ shape_x,
                                   const uint32_t* __restrict__ strides_x,
                                   const uint32_t* __restrict__ shape_z,
                                   const uint32_t* __restrict__ strides_z,
                                   const {{ dtype }} a,
                                   const uint32_t ndim,
                                   const uint32_t n){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        uint32_t index_x = index_transform(i, strides_x, shape_x, ndim);
        uint32_t index_z = index_transform(i, strides_z, shape_z, ndim);

        z[index_z] = x[index_x] / a;
    }
}
