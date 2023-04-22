
#include <stdint.h>

__global__ void normalize_tf_{{ intype }}_{{ outtype }}(const {{ intype }}* __restrict__ src_img,
                          {{ outtype }}* __restrict__ dst_img,
                          uint16_t width,
                          uint16_t height,
                          uint8_t channels){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {

        uint32_t offset = (j   * (width) + i  ) * channels;

        for (uint8_t c = 0; c < channels; c++){

            dst_img[offset + c] = ({{ outtype }}) src_img[offset + c] / 127.5 - 1.0;

        }
    }
}

__global__ void CHW_normalize_mean_std_{{ intype }}_{{ outtype }}({{ intype }}* __restrict__ src_img,
                          {{ outtype }}* __restrict__ dst_img,
                          uint16_t width,
                          uint16_t height,
                          uint8_t channels,
                          float* __restrict__ mean,
                          float* __restrict__ std){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {

        uint32_t offset = (j * width + i);
        uint32_t total_size = height * width;
        for (uint8_t c = 0; c < channels; c++){
            dst_img[c * total_size + offset] = ({{ outtype }}) (({{ outtype }})  (src_img[c * total_size + offset] / 255.0) - mean[c] ) / std[c];
        }
    }
}

__global__ void HWC_normalize_mean_std_{{ intype }}_{{ outtype }}(const {{ intype }}* __restrict__ src_img,
                          {{ outtype }}* __restrict__ dst_img,
                          uint16_t width,
                          uint16_t height,
                          uint8_t channels,
                          float* __restrict__ mean,
                          float* __restrict__ std){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {

        uint32_t offset = (j   * (width) + i  ) * channels;

        for (uint8_t c = 0; c < channels; c++){

            dst_img[offset + c] = ({{ outtype }}) (({{ outtype }}) (src_img[offset + c] / 255.0) - mean[c] ) / std[c];

        }
    }
}

__global__ void normalize_255_{{ intype }}_{{ outtype }}(const {{ intype }}* __restrict__ src_img,
                          {{ outtype }}* __restrict__ dst_img,
                          uint16_t width,
                          uint16_t height,
                          uint8_t channels){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {

        uint32_t offset = (j   * (width) + i  ) * channels;

        for (uint8_t c = 0; c < channels; c++){

            dst_img[offset + c] = ({{ outtype }}) ({{ outtype }}) (src_img[offset + c] / 255.0);

        }
    }
}
