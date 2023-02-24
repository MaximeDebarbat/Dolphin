#include <stdint.h>

struct ImageSize {
    uint16_t width;
    uint16_t height;
    uint16_t channels;
};

__global__ void resize(uint8_t *src_img, 
                       uint8_t *dst_img,
                       ImageSize *src_image_size,
                       ImageSize *dst_image_size){

    /*
    src_img and dst_img are expected to be HWC images     
    */
    //printf("src_image_size : (%d,%d,%d) dst_image_size : (%d,%d,%d)\n",src_image_size->width,src_image_size->height,src_image_size->channels,dst_image_size->width,dst_image_size->height,dst_image_size->channels);

    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < (dst_image_size->width) && j < (dst_image_size->height)) 
    {

        //printf("(%d,%d) \n",i, j);
        uint32_t iIn = (uint32_t) (i * (src_image_size->width) / (dst_image_size->width));
        uint32_t jIn = (uint32_t) (j * (src_image_size->height) / (dst_image_size->height));

        uint32_t dst_offset = (j   * (dst_image_size->width) + i  ) * (src_image_size->channels);
        uint32_t src_offset = (jIn * (src_image_size->width) + iIn) * (src_image_size->channels);

        for (uint32_t c = 0; c < src_image_size->channels; ++c) 
        { 
            //printf("(%d,%d) -> %d\n",i,j,src_img[(uint32_t) src_offset + c]);
            dst_img[(uint32_t) dst_offset + c] = src_img[(uint32_t) src_offset + c];
        }
    }
}