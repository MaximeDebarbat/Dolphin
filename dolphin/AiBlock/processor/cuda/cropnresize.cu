
#include <stdint.h>

struct ImageDimension {
    uint16_t height;
    uint16_t width;
    uint16_t channels;
};

struct BoundingBox{
    uint16_t x0;
    uint16_t y0;
    uint16_t x1;
    uint16_t y1;
};

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

__global__ void cropnresize(uint8_t *src_image,
                            uint8_t* dst_images,
                            ImageDimension* in_size,
                            ImageDimension* out_size,
                            BoundingBox* bounding_boxes
                            ){

    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t j = blockDim.y * blockIdx.y + threadIdx.y;
    uint16_t b = blockDim.z * blockIdx.z + threadIdx.z;

    uint16_t bbox_width = bounding_boxes[b].x1 - bounding_boxes[b].x0;
    uint16_t bbox_height = bounding_boxes[b].y1 - bounding_boxes[b].y0;

    if (i < (out_size->width) && j < (out_size->height) && bbox_width > 0 && bbox_height > 0)
    {

        uint32_t iIn = (uint32_t) bounding_boxes[b].x0+(i * (bbox_width) / (out_size->width));
        uint32_t jIn = (uint32_t) bounding_boxes[b].y0+(j * (bbox_height) / (out_size->height));

        uint32_t dst_offset =  (j   * (out_size->width) + i  ) * (out_size->channels) + b*out_size->width*out_size->height*out_size->channels;
        uint32_t src_offset =  (jIn * (in_size->width)  + iIn) * (in_size->channels);

        for (uint8_t c = 0; c < in_size->channels; ++c)
        {
            dst_images[(uint32_t) dst_offset + c] = src_image[(uint32_t) src_offset + c];
        }
    }

}

__global__ void getmaxdim(BoundingBox *bounding_boxes,
                          float *max_width,
                          float *max_height){

    uint16_t index = threadIdx.x;

    uint16_t local_max_width = bounding_boxes[index].x1 - bounding_boxes[index].x0;
    uint16_t local_max_height = bounding_boxes[index].y1 - bounding_boxes[index+1].y0;

    atomicMaxFloat(max_width, local_max_width);
    atomicMaxFloat(max_height, local_max_height);
}