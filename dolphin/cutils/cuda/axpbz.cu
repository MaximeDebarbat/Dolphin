
#include <stdint.h>


__global__ void axpbz_{{ dtype }}(const {{ dtype }}* __restrict__ x,
                                  {{ dtype }}* __restrict__  z,
                                  const {{ dtype }} a,
                                  const {{ dtype }} b,
                                  const uint32_t* __restrict__  shape,
                                  const uint32_t* __restrict__  strides,
                                  const uint32_t ndim,
                                  uint32_t n){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        uint32_t index = index_transform(i, strides, shape, ndim);
        z[index] = a*x[index] + b;
    }
}
