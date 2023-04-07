
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