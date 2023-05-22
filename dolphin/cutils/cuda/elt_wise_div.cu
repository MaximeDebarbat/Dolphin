
#include <stdint.h>
#include <assert.h>

__global__ void elt_wise_div_{{ dtype }}(const {{ dtype }}* __restrict__ x,
                                         const {{ dtype }}* __restrict__ y,
                                   {{ dtype }}* __restrict__ z,
                                   const uint32_t* __restrict__ shape_x,
                                  const uint32_t* __restrict__ strides_x,
                                  const uint32_t* __restrict__ shape_y,
                                  const uint32_t* __restrict__ strides_y,
                                  const uint32_t* __restrict__ shape_z,
                                  const uint32_t* __restrict__ strides_z,
                                  const uint32_t ndim,
                                  const uint32_t n,
                                   uint8_t *error){

    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){

        uint32_t index_x = index_transform(i, strides_x, shape_x, ndim);
        uint32_t index_y = index_transform(i, strides_y, shape_y, ndim);
        uint32_t index_z = index_transform(i, strides_z, shape_z, ndim);

        if(y[index_y] == 0){
            z[index_z] = ({{dtype}}) 0;
            *error = 1;
            return;
        }

        z[index_z] = x[index_x] / y[index_y];
    }
}
