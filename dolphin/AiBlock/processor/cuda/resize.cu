#include <stdint.h>

struct ImageSize {
    uint16_t width;
    uint16_t height;
    uint16_t channels;
};

__global__ void resize(uint16_t *src_img, 
                       uint16_t *dst_img,
                       ImageSize *src_image_size,
                       ImageSize *dst_image_size){

    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < (*dst_w) && j < (*dst_h)) 
    {
        uint32_t iIn = (uint32_t) (i * (src_image_size->width) / (dst_image_size->width));
        uint32_t jIn = (uint32_t) (j * (src_image_size->height) / (dst_image_size->height));

        uint32_t dst_offset = (j   * (dst_image_size->width) + i  ) * (src_image_size->channels);
        uint32_t src_offset = (jIn * (src_image_size->width) + iIn) * (src_image_size->channels);

        for (uint32_t c = 0; c < src_image_size->channels; ++c) 
        { 
            dst_img[(uint32_t) dst_offset + c] = src_img[(uint32_t) src_offset + c];
        }
    }
}