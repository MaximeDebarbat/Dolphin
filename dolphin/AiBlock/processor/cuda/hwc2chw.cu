
#include <stdint.h>

struct ImageDimension {
    uint16_t height;
    uint16_t width;
    uint16_t channels;
};

__global__ void hwc2chw(ImageDimension *im_size,
                          uint8_t *in_image,
                          uint8_t *out_image){

    uint16_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < im_size->width && j < im_size->height) {

        uint32_t src_offset = (j * im_size->width + i) * im_size->channels;
        uint32_t dst_offset = (j * im_size->width + i);

        for (uint16_t c = 0; c < im_size->channels; c++) {

            out_image[c *(im_size->height * im_size->width) + dst_offset] = in_image[src_offset + c];
        }
    }

}