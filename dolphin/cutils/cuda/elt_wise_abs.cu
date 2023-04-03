
#include <stdint.h>

__global__ void elt_wise_abs_{{dtype}}({{dtype}}* in, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    in[i] = in[i] > 0 ? in[i] : -in[i];
  }
}