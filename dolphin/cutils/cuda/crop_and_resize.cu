
#include <stdint.h>

    /*
        /!\ WARNING /!\
            Bounding Boxes have to be particularly formatted.
            (n,4) in order for the data in memory to look that way
            [x0, y0, x1, y1, x0, y0, x1, y1, ...]
            |----- n0 -----|----- n1 -----|  ...  |

        /!\ Note /!\
            Bounding box is accessible that way.
            bounding_boxes[n][0] -> x0
            bounding_boxes[n][1] -> y0
            bounding_boxes[n][2] -> x1
            bounding_boxes[n][3] -> y1

            or
            bounding_boxes[n] -> x0
            bounding_boxes[n+1] -> y0
            bounding_boxes[n+2] -> x1
            bounding_boxes[n+3] -> y1
    */

#ifndef BBOX_STRUCT
#define BBOX_STRUCT

struct BoundingBox{
    uint32_t x0;
    uint32_t y0;
    uint32_t x1;
    uint32_t y1;
};

#endif

/*

                NEAREST HWC CROP N RESIZE

*/
__global__ void HWC_crop_and_resize_{{ dtype }}(
                const {{ dtype }}* __restrict__ src_images,
                {{ dtype }}* __restrict__ dst_images,
                // Bounding Boxes
                const BoundingBox* bounding_boxes,
                const uint32_t n_bounding_boxes,
                // Resolution
                const uint32_t width,
                const uint32_t height,
                // Array params
                const uint32_t* __restrict__  shape_src,
                const uint32_t* __restrict__  strides_src,
                const uint32_t ndim
                ){

    uint32_t b = blockDim.z * blockIdx.z + threadIdx.z;

    if(b >= n_bounding_boxes){
        return;
    }

    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    const uint32_t in_width = shape_src[1];
    const uint32_t in_channels = ndim == 3 ? shape_src[2] : 1;

    const uint32_t bbox_width = bounding_boxes[b].x1 - bounding_boxes[b].x0;
    const uint32_t bbox_height = bounding_boxes[b].y1 - bounding_boxes[b].y0;

    if(i < width && j < height && bbox_width > 0 && bbox_height > 0){

        uint32_t iIn = (uint32_t) bounding_boxes[b].x0+(i * (bbox_width) / (width));
        uint32_t jIn = (uint32_t) bounding_boxes[b].y0+(j * (bbox_height) / (height));

        uint32_t dst_offset =  (j   * (width) + i  ) * in_channels + b*width*height*in_channels;
        uint32_t src_offset =  (jIn * (in_width)  + iIn) * in_channels;

        for (uint32_t c = 0; c < (in_channels); ++c)
        {
            dst_images[dst_offset + c] =
            src_images[index_transform(src_offset + c, strides_src, shape_src, ndim)];
        }
    }
}

/*

                NEAREST CHW CROP N RESIZE

*/
__global__ void CHW_crop_and_resize_{{ dtype }}(
                const {{ dtype }}* __restrict__ src_images,
                {{ dtype }}* __restrict__ dst_images,
                // Bounding Boxes
                const BoundingBox* bounding_boxes,
                const uint32_t n_bounding_boxes,
                // Resolution
                const uint32_t width,
                const uint32_t height,
                // Array params
                const uint32_t* __restrict__  shape_src,
                const uint32_t* __restrict__  strides_src,
                const uint32_t ndim
                ){

    uint32_t b = blockDim.z * blockIdx.z + threadIdx.z;

    if(b >= n_bounding_boxes){
        return;
    }

    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    const uint32_t in_channels = shape_src[0];
    const uint32_t in_height = shape_src[1];
    const uint32_t in_width = shape_src[2];

    const uint32_t bbox_width = bounding_boxes[b].x1 - bounding_boxes[b].x0;
    const uint32_t bbox_height = bounding_boxes[b].y1 - bounding_boxes[b].y0;

    uint32_t total_size_out = (width*height);
    uint32_t total_size_in = (in_width*in_height);

    if(i < width && j < height && bbox_width > 0 && bbox_height > 0){

        uint32_t iIn = (uint32_t) bounding_boxes[b].x0+(i * (bbox_width) / (width));
        uint32_t jIn = (uint32_t) bounding_boxes[b].y0+(j * (bbox_height) / (height));

        uint32_t dst_offset =  (j   * (width) + i  ) + b*width*height*in_channels;
        uint32_t src_offset =  (jIn * (in_width)  + iIn);

        for (uint32_t c = 0; c < (in_channels); ++c)
        {
            dst_images[dst_offset + total_size_out * c] =
            src_images[index_transform(src_offset + total_size_in * c, strides_src, shape_src, ndim)];
        }
    }
}

/*

                PADDED HWC CROP N RESIZE

*/
__global__ void HWC_crop_and_resize_padding_{{ dtype }}(
                const {{ dtype }}* __restrict__ src_images,
                {{ dtype }}* __restrict__ dst_images,
                // Bounding Boxes
                const BoundingBox* bounding_boxes,
                const uint32_t n_bounding_boxes,
                // Resolution
                const uint32_t width,
                const uint32_t height,
                // Array params
                const uint32_t* __restrict__  shape_src,
                const uint32_t* __restrict__  strides_src,
                const uint32_t ndim,
                {{ dtype }} padding_value
                ){

    uint32_t b = blockDim.z * blockIdx.z + threadIdx.z;

    if(b >= n_bounding_boxes){
        return;
    }

    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    const uint32_t in_width = shape_src[1];
    const uint32_t in_channels = ndim == 3 ? shape_src[2] : 1;

    const uint32_t bbox_width = bounding_boxes[b].x1 - bounding_boxes[b].x0;
    const uint32_t bbox_height = bounding_boxes[b].y1 - bounding_boxes[b].y0;

    if(i < width && j < height && bbox_width > 0 && bbox_height > 0){

        float scale = (float) width / bbox_width > (float) height / bbox_height ?
                        (float) height / bbox_height :
                        (float) width / bbox_width;

        uint32_t true_width = (uint32_t) bbox_width * scale;
        uint32_t true_height = (uint32_t) bbox_height * scale;

        uint32_t x0 = (uint32_t) (width - true_width) / 2;
        uint32_t y0 = (uint32_t) (height - true_height) / 2;
        uint32_t x1 = (uint32_t) x0 + true_width;
        uint32_t y1 = (uint32_t) y0 + true_height;

        uint32_t dst_offset =  (j * (width) + i ) * in_channels + b*width*height*in_channels;

        if(i<x0 || i>=x1 || j<y0 || j>=y1){

            for(uint32_t k=0; k<in_channels; k++){
                dst_images[(uint32_t) dst_offset + k] = padding_value;
            }

        }else{

            uint32_t iIn = (uint32_t) bounding_boxes[b].x0 + (i - x0) * (bbox_width) / (true_width);
            uint32_t jIn = (uint32_t) bounding_boxes[b].y0 + (j - y0) * (bbox_height) / (true_height);

            uint32_t src_offset =  (jIn * (in_width)  + iIn) * in_channels;

            for (uint32_t c = 0; c < (in_channels); ++c)
            {
                dst_images[dst_offset + c] =
                src_images[index_transform(src_offset + c, strides_src, shape_src, ndim)];
            }

        }
    }
}

/*

                PADDED CHW CROP N RESIZE

*/
__global__ void CHW_crop_and_resize_padding_{{ dtype }}(
                const {{ dtype }}* __restrict__ src_images,
                {{ dtype }}* __restrict__ dst_images,
                // Bounding Boxes
                const BoundingBox* bounding_boxes,
                const uint32_t n_bounding_boxes,
                // Resolution
                const uint32_t width,
                const uint32_t height,
                // Array params
                const uint32_t* __restrict__  shape_src,
                const uint32_t* __restrict__  strides_src,
                const uint32_t ndim,
                {{ dtype }} padding_value
                ){

    uint32_t b = blockDim.z * blockIdx.z + threadIdx.z;

    if(b >= n_bounding_boxes){
        return;
    }

    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    const uint32_t in_height = shape_src[1];
    const uint32_t in_width = shape_src[2];
    const uint32_t in_channels = ndim == 3 ? shape_src[0] : 1;

    const uint32_t bbox_width = bounding_boxes[b].x1 - bounding_boxes[b].x0;
    const uint32_t bbox_height = bounding_boxes[b].y1 - bounding_boxes[b].y0;

    if(i < width && j < height && bbox_width > 0 && bbox_height > 0){

        float scale = (float) width / bbox_width > (float) height / bbox_height ?
                        (float) height / bbox_height :
                        (float) width / bbox_width;

        uint32_t true_width = (uint32_t) bbox_width * scale;
        uint32_t true_height = (uint32_t) bbox_height * scale;

        uint32_t x0 = (uint32_t) (width - true_width) / 2;
        uint32_t y0 = (uint32_t) (height - true_height) / 2;
        uint32_t x1 = (uint32_t) x0 + true_width;
        uint32_t y1 = (uint32_t) y0 + true_height;

        uint32_t dst_offset =  (j   * (width) + i  ) + b*width*height*in_channels;
        uint32_t total_size_out = (width*height);
        uint32_t total_size_in = (in_width*in_height);

        if(i<x0 || i>=x1 || j<y0 || j>=y1){

            for(uint32_t k=0; k<in_channels; k++){
                dst_images[(uint32_t) dst_offset + total_size_out * k] = padding_value;
            }

        }else{

            uint32_t iIn = (uint32_t) bounding_boxes[b].x0 + (i - x0) * (bbox_width) / (true_width);
            uint32_t jIn = (uint32_t) bounding_boxes[b].y0 + (j - y0) * (bbox_height) / (true_height);

            uint32_t src_offset =  (jIn * (in_width)  + iIn);

            for (uint32_t c = 0; c < (in_channels); ++c)
            {
                dst_images[dst_offset + total_size_out * c] =
                src_images[index_transform(src_offset + total_size_in * c, strides_src, shape_src, ndim)];
            }
        }
    }
}