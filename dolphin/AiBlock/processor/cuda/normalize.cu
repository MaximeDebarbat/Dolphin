
#include <stdint.h>

struct ImageSize {
    uint16_t height;
    uint16_t width;
    uint16_t channels;
};

__global__ void normalize_255(uint8_t *src_img,
                          float *dst_img,
                          ImageSize *image_size){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < image_size->width && j < image_size->height) {
        uint32_t offset = (j * image_size->width + i) * image_size->channels;

        for (uint16_t c = 0; c < image_size->channels; c++){

            dst_img[offset + c] = (float) src_img[offset + c] / 255.0;

        }
    }

}

__global__ void normalize_128(uint8_t *src_img,
                          float *dst_img,
                          ImageSize *image_size){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < image_size->width && j < image_size->height) {
        uint32_t offset = (j * image_size->width + i) * image_size->channels;

        for (uint16_t c = 0; c < image_size->channels; c++){

            dst_img[offset + c] = (float) ((float) src_img[offset + c] - 128.0) / 128.0;

        }
    }

}

__global__ void normalize_mean_std(uint8_t *src_img,
                          float *dst_img,
                          ImageSize *image_size,
                          float* mean,
                          float* std){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < image_size->width && j < image_size->height) {

        uint32_t offset = (j * image_size->width + i);

        for (uint16_t c = 0; c < image_size->channels; c++){

            dst_img[c *(image_size->height * image_size->width) + offset] = (float) ((float) (src_img[c *(image_size->height * image_size->width) + offset] / 255.0) - mean[c] ) / std[c];

        }
    }

}