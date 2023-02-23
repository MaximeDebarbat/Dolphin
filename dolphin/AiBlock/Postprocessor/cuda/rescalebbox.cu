
#include <stdint.h>

struct ImageSize {
    uint16_t width;
    uint16_t height;
    uint16_t channels;
};

struct BoundingBox{
    uint16_t x0;
    uint16_t y0;
    uint16_t x1;
    uint16_t y1;
};

__global__ void rescalebbox(BoundingBox *bboxes,
                            BoundingBox *out_bboxes,
                            ImageSize *original_image_size,
                            ImageSize *rescaled_image_size){

    uint16_t index = threadIdx.x;

    double x0 = (double) bboxes[index].x0/rescaled_image_size->width;
    double y0 = (double) bboxes[index].y0/rescaled_image_size->height;
    double x1 = (double) bboxes[index].x1/rescaled_image_size->width;
    double y1 = (double) bboxes[index].y1/rescaled_image_size->height;

    out_bboxes[index].x0 = (uint16_t) original_image_size->width*x0;
    out_bboxes[index].y0 = (uint16_t) original_image_size->height*y0;
    out_bboxes[index].x1 = (uint16_t) original_image_size->width*x1;
    out_bboxes[index].y1 = (uint16_t) original_image_size->height*y1;

}