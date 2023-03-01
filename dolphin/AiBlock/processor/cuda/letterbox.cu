

#include <stdint.h>

struct ImageSize {
    uint16_t height;
    uint16_t width;
    uint16_t channels;
};

__global__ void letterbox(uint8_t *src_img, 
                          uint8_t *dst_img,
                          ImageSize *src_image_size,
                          ImageSize *dst_image_size,
                          uint8_t *padding_value){

    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t j = blockDim.y * blockIdx.y + threadIdx.y;
    

    if (i < (dst_image_size->width) && j < (dst_image_size->height)) 
    {

        float scale = (float) dst_image_size->width / src_image_size->width > (float) dst_image_size->height / src_image_size->height ? 
                      (float) dst_image_size->height / src_image_size->height : 
                      (float) dst_image_size->width / src_image_size->width;

        uint16_t true_width = (uint16_t) src_image_size->width * scale;
        uint16_t true_height = (uint16_t) src_image_size->height * scale;
        
        uint16_t x0 = (uint16_t) (dst_image_size->width - true_width) / 2;
        uint16_t y0 = (uint16_t) (dst_image_size->height - true_height) / 2;
        uint16_t x1 = (uint16_t) x0 + true_width;
        uint16_t y1 = (uint16_t) y0 + true_height;

        uint32_t dst_offset =  (j   * (dst_image_size->width) + i  ) * (dst_image_size->channels);

        if(i<x0 || i>=x1 || j<y0 || j>=y1){
            for(uint32_t k=0; k<dst_image_size->channels; k++){
                dst_img[(uint32_t) dst_offset + k] = *padding_value;
            }
        }
        else{

            uint32_t iIn = (uint32_t) (i - x0) * (src_image_size->width) / (true_width);
            uint32_t jIn = (uint32_t) (j - y0) * (src_image_size->height) / (true_height);

            uint32_t src_offset = (jIn * (src_image_size->width)  + iIn) * (src_image_size->channels);

            for(uint16_t k=0; k<dst_image_size->channels; k++){
                dst_img[(uint32_t) dst_offset + k] = src_img[(uint32_t) src_offset + k];
            }
        }

    }
}