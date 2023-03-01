
#include <stdint.h>

struct ImageSize {
    uint16_t height;
    uint16_t width;
    uint16_t channels;
};

__global__ void bgr2rgb(ImageSize *im_size,
                          uint8_t *image){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < im_size->width && y < im_size->height){

        uint32_t offset = (y * im_size->width + x) * im_size->channels;

        uint8_t temp = image[offset];
        image[offset] = image[offset+2];
        image[offset+2] = temp;

    }

}