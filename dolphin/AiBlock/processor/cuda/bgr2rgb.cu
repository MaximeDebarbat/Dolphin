
#include <stdint.h>

struct ImageDimension {
    uint16_t height;
    uint16_t width;
    uint16_t channels;
};

__global__ void bgr2rgb(ImageDimension *im_size,
                          uint8_t *in_image,
                          uint8_t *out_image){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < im_size->width && y < im_size->height){

        uint32_t offset = (y * im_size->width + x) * im_size->channels;

        uint8_t temp = in_image[offset];
        out_image[offset] = in_image[offset+2];
        out_image[offset+2] = temp;

    }

}