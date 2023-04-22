
#include <stdint.h>

__global__ void elt_wise_abs_{{dtype}}({{dtype}}* in, {{dtype}}* out, uint32_t n) {

  for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    out[i] = in[i] > 0 ? in[i] : -in[i];
  }
}