
#include <stdint.h>

/*
    Not optimized for speed, just for clarity.
    TODO:
     - Gride-stride loop
     - Shared memory
*/

__global__ void HWC_resize_nearest_{{ dtype }}(
    const {{ dtype }}* __restrict__ src_img,
    {{ dtype }}* __restrict__ dst_img,
    const uint16_t input_width,
    const uint16_t input_height,
    const uint16_t output_width,
    const uint16_t output_height,
    const uint8_t channels
){

    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < (output_width) && j < (output_height))
    {

        uint32_t iIn = (uint32_t) (i * (input_width) / (output_width));
        uint32_t jIn = (uint32_t) (j * (input_height) / (output_height));

        uint32_t dst_offset = (j   * (output_width) + i  ) * (channels);
        uint32_t src_offset = (jIn * (input_width) + iIn) * (channels);

        for (uint8_t c = 0; c < channels; ++c)
        {
            dst_img[(uint32_t) dst_offset + c] = src_img[(uint32_t) src_offset + c];
        }
    }

}

__global__ void CHW_resize_nearest_{{ dtype }}(
    const {{ dtype }}* __restrict__ src_img,
    {{ dtype }}* __restrict__ dst_img,
    const uint16_t input_width,
    const uint16_t input_height,
    const uint16_t output_width,
    const uint16_t output_height,
    const uint8_t channels
){

    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < (output_width) && j < (output_height))
    {

        uint32_t iIn = (uint32_t) (i * (input_width) / (output_width));
        uint32_t jIn = (uint32_t) (j * (input_height) / (output_height));

        uint32_t dst_offset = (j   * (output_width) + i  );
        uint32_t src_offset = (jIn * (input_width) + iIn);

        uint32_t total_size_out = (output_width*output_height);
        uint32_t total_size_in = (input_width*input_height);

        for (uint8_t c = 0; c < channels; ++c)
        {

            dst_img[(uint32_t) dst_offset + total_size_out * c] = src_img[(uint32_t) src_offset + total_size_in * c];
        }
    }

}

__global__ void HWC_resize_padding_{{ dtype }}(
    const {{ dtype }}* __restrict__ src_img,
    {{ dtype }}* __restrict__ dst_img,
    const uint16_t input_width,
    const uint16_t input_height,
    const uint16_t output_width,
    const uint16_t output_height,
    const uint8_t channels,
    const {{ dtype }} padding_value
){

    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < (output_width) && j < (output_height))
    {

        float scale = (float) output_width / input_width > (float) output_height / input_height ?
                      (float) output_height / input_height :
                      (float) output_width / input_width;

        uint16_t true_width = (uint16_t) input_width * scale;
        uint16_t true_height = (uint16_t) input_height * scale;

        uint16_t x0 = (uint16_t) (output_width - true_width) / 2;
        uint16_t y0 = (uint16_t) (output_height - true_height) / 2;
        uint16_t x1 = (uint16_t) x0 + true_width;
        uint16_t y1 = (uint16_t) y0 + true_height;

        uint32_t dst_offset =  (j   * (output_width) + i  ) * channels;

        if(i<x0 || i>=x1 || j<y0 || j>=y1){
            for(uint32_t k=0; k<channels; k++){
                dst_img[(uint32_t) dst_offset + k] = padding_value;
            }
        }else{

            uint32_t iIn = (uint32_t) (i - x0) * (input_width) / (true_width);
            uint32_t jIn = (uint32_t) (j - y0) * (input_height) / (true_height);

            uint32_t src_offset = (jIn * (input_width)  + iIn) * channels;

            for(uint16_t k=0; k<channels; k++){
                dst_img[(uint32_t) dst_offset + k] = src_img[(uint32_t) src_offset + k];
            }
        }

    }
}

__global__ void CHW_resize_padding_{{ dtype }}(
    const {{ dtype }}* __restrict__ src_img,
    {{ dtype }}* __restrict__ dst_img,
    const uint16_t input_width,
    const uint16_t input_height,
    const uint16_t output_width,
    const uint16_t output_height,
    const uint8_t channels,
    const {{ dtype }} padding_value
){

    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t j = blockDim.y * blockIdx.y + threadIdx.y;


    if (i < (output_width) && j < (output_height))
    {

        float scale = (float) output_width / input_width > (float) output_height / input_height ?
                      (float) output_height / input_height :
                      (float) output_width / input_width;

        uint16_t true_width = (uint16_t) input_width * scale;
        uint16_t true_height = (uint16_t) input_height * scale;

        uint16_t x0 = (uint16_t) (output_width - true_width) / 2;
        uint16_t y0 = (uint16_t) (output_height - true_height) / 2;
        uint16_t x1 = (uint16_t) x0 + true_width;
        uint16_t y1 = (uint16_t) y0 + true_height;

        uint32_t dst_offset = j * (output_width) + i;
        uint32_t total_size_out = (output_width) * (output_height);
        uint32_t total_size_in = (input_width) * (input_height);

        if(i<x0 || i>=x1 || j<y0 || j>=y1){
            for(uint32_t k=0; k<channels; k++){
                dst_img[(uint32_t) dst_offset + total_size_out * k] = padding_value;
            }
        }else{

            uint32_t iIn = (uint32_t) (i - x0) * (input_width) / (true_width);
            uint32_t jIn = (uint32_t) (j - y0) * (input_height) / (true_height);

            uint32_t src_offset = (jIn * (input_width)  + iIn);

            for(uint16_t k=0; k<channels; k++){
                dst_img[(uint32_t) dst_offset + total_size_out*k] = src_img[(uint32_t) src_offset + total_size_in*k];
            }
        }

    }
}