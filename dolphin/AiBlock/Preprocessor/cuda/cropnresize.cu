
#include <stdint.h>

__device__ __forceinline__ float atomicMaxFloat(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void cropnresize(uint16_t *src_image, 
                            uint16_t* dst_images,
                            uint16_t* in_size,
                            uint16_t* out_size,
                            float** bounding_boxes
                            ){

    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t j = blockDim.y * blockIdx.y + threadIdx.y;
    uint16_t b = blockDim.z * blockIdx.z + threadIdx.z;

    uint16_t in_channels = in_size[0];
    uint16_t in_width = in_size[1];
    uint16_t in_height = in_size[2];

    uint16_t out_width = out_size[1];
    uint16_t out_height = out_size[2];
    
    const uint16_t current_width = (uint16_t) out_width *   (bounding_boxes[b][2] - bounding_boxes[b][0]); /* x1 - x0 */
    const uint16_t current_height = (uint16_t) out_height * (bounding_boxes[b][3] - bounding_boxes[b][1]); /* y1 - y0 */

    const uint16_t x = (uint16_t) i + bounding_boxes[b][0];
    const uint16_t y = (uint16_t) j * bounding_boxes[b][1];

    if(i<current_width && j<current_height){

        uint16_t iIn = (uint16_t) (x * (in_width) / (out_width));
        uint16_t jIn = (uint16_t) (y * (out_width) / (out_height));

        uint32_t dst_offset = (j   * (out_height) + i  ) * (in_channels);
        uint32_t src_offset = (jIn * (out_width) + iIn) * (in_channels);

        for (uint16_t c = 0; c < (in_channels); ++c) 
        { 
            dst_images[(uint16_t) dst_offset + c] = src_image[(uint16_t) src_offset + c];
        }
    }
}

__global__ void getmaxdim(float *bounding_boxes, 
                          uint16_t *size,
                          float *max_width,
                          float *max_height){

    uint16_t index = threadIdx.x;

    //printf("%d -> %.2f\n",index, bounding_boxes[index]);
    //printf("%d %d \n", index, *size);

    if (index<*size) {

        index*=4;

        float local_max_width = bounding_boxes[index+2] - bounding_boxes[index];
        float local_max_height = bounding_boxes[index+3] - bounding_boxes[index+1];

        //printf("( %.2f , %.2f )\n",local_max_width, local_max_height);

        atomicMaxFloat(max_width, local_max_width);
        atomicMaxFloat(max_height, local_max_height);

    }

    //__syncthreads();
    //printf("AtomicMax (%.2f, %.2f)\n", *max_width, *max_height);
    //__syncthreads();

}