
#include <stdint.h>

__device__ __forceinline__ uint32_t index_transform(const uint32_t index,
                                                      const uint32_t* __restrict__ strides,
                                                      const uint32_t* __restrict__ shape,
                                                      const uint32_t ndim
                                                     ){

    uint32_t idx = index;
    uint32_t res = 0;

    for(uint32_t __i = 0; __i < ndim; __i++){
        uint32_t r_i = ndim - __i - 1;
        res += (idx % shape[r_i]) * strides[r_i];
        idx /= shape[r_i];
    }
    return res;
}
