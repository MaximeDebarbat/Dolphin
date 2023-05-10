
#include <stdint.h>

__global__ void elt_wise_abs_{{dtype}}(const {{dtype}}* __restrict__ in,
                                       {{dtype}}* __restrict__ out,
                                       const uint32_t* __restrict__ shape,
                                       const uint32_t* __restrict__ strides,
                                       const uint32_t ndim,
                                       uint32_t n) {

  for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    uint32_t index_copy = i;
    uint32_t index = 0;

    for(uint32_t __i = 0; __i < ndim; __i++){
        uint32_t r_i = ndim - __i - 1;
        index += (index_copy % shape[r_i]) * strides[r_i];
        index_copy /= shape[r_i];
    }

    out[index] = in[index] > 0 ? in[index] : -in[index];
  }
}