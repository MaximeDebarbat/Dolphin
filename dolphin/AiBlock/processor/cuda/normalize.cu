
#include <stdint.h>

struct ImageSize {
    uint16_t height;
    uint16_t width;
    uint16_t channels;
};

__global__ void normalize(uint8_t *src_img,
                          float *dst_img,
                          ImageSize *image_size){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < image_size->width && j < image_size->height) {
        //printf("width, height : %d, %d \n", image_size->width, image_size->height);
        uint32_t offset = (j * image_size->width + i) * image_size->channels;

        for (uint16_t c = 0; c < image_size->channels; c++){

            dst_img[offset + c] = (float) src_img[offset + c] / 255.0;

        }
    }

}