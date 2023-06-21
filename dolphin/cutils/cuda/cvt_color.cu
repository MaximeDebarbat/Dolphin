
#include <stdint.h>

__global__ void HWC_cvt_color_rgb2gray_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint32_t* __restrict__  shape_src,
    const uint32_t* __restrict__  strides_src,
    const uint32_t* __restrict__  shape_dst,
    const uint32_t* __restrict__  strides_dst,
    const uint32_t ndim_src,
    const uint32_t ndim_dst) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        uint32_t channels = ndim_src == 2 ? 1 : shape_src[2];
        const uint32_t src_idx = (j * width + i) * channels;
        const uint32_t dst_idx = index_transform(j * width + i, strides_dst, shape_dst, ndim_dst);
        float r = src[index_transform(src_idx, strides_src, shape_src, ndim_src)];
        float g = src[index_transform(src_idx + 1, strides_src, shape_src, ndim_src)];
        float b = src[index_transform(src_idx + 2, strides_src, shape_src, ndim_src)];
        dst[dst_idx] = ({{ dtype }})(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

__global__ void CHW_cvt_color_rgb2gray_{{ dtype }}(
    const {{ dtype  }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint32_t* __restrict__  shape_src,
    const uint32_t* __restrict__  strides_src,
    const uint32_t* __restrict__  shape_dst,
    const uint32_t* __restrict__  strides_dst,
    const uint32_t ndim_src,
    const uint32_t ndim_dst) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = j * width + i;
        {{ dtype }} r = src[index_transform(idx, strides_src, shape_src, ndim_src)]; // To fix
        {{ dtype }} g = src[index_transform(idx + width * height, strides_src, shape_src, ndim_src)]; // To fix
        {{ dtype }} b = src[index_transform(idx + 2 * width * height, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx, strides_dst, shape_dst, ndim_dst)] = ({{ dtype }})(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

__global__ void HWC_cvt_color_bgr2gray_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint32_t* __restrict__  shape_src,
    const uint32_t* __restrict__  strides_src,
    const uint32_t* __restrict__  shape_dst,
    const uint32_t* __restrict__  strides_dst,
    const uint32_t ndim_src,
    const uint32_t ndim_dst) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        uint32_t channels = ndim_src == 2 ? 1 : shape_src[2];
        const uint32_t src_idx = (j * width + i) * channels;
        const uint32_t dst_idx = j * width + i;
        {{ dtype }} r = src[index_transform(src_idx + 2, strides_src, shape_src, ndim_src)]; // To fix
        {{ dtype }} g = src[index_transform(src_idx + 1, strides_src, shape_src, ndim_src)]; // To fix
        {{ dtype }} b = src[index_transform(src_idx, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(dst_idx, strides_dst, shape_dst, ndim_dst)] = ({{ dtype }})(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

__global__ void CHW_cvt_color_bgr2gray_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint32_t* __restrict__  shape_src,
    const uint32_t* __restrict__  strides_src,
    const uint32_t* __restrict__  shape_dst,
    const uint32_t* __restrict__  strides_dst,
    const uint32_t ndim_src,
    const uint32_t ndim_dst) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = j * width + i;
        {{ dtype }} r = src[index_transform(idx + 2 * width * height, strides_src, shape_src, ndim_src)]; // To fix
        {{ dtype }} g = src[index_transform(idx + width * height, strides_src, shape_src, ndim_src)]; // To fix
        {{ dtype }} b = src[index_transform(idx, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx, strides_dst, shape_dst, ndim_dst)] = ({{ dtype }})(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

__global__ void HWC_cvt_color_rgb2bgr_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint32_t* __restrict__  shape_src,
    const uint32_t* __restrict__  strides_src,
    const uint32_t* __restrict__  shape_dst,
    const uint32_t* __restrict__  strides_dst,
    const uint32_t ndim_src,
    const uint32_t ndim_dst) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        uint32_t channels = ndim_src == 2 ? 1 : shape_src[2];
        const uint32_t idx = (j * width + i) * channels;
        {{ dtype }} tmp = src[index_transform(idx, strides_src, shape_src, ndim_src)]; // In case src == dst
        dst[index_transform(idx, strides_dst, shape_dst, ndim_dst)] = src[index_transform(idx + 2, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx + 1, strides_dst, shape_dst, ndim_dst)] = src[index_transform(idx + 1, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx + 2, strides_dst, shape_dst, ndim_dst)] = tmp; // To fix
    }
}

__global__ void CHW_cvt_color_rgb2bgr_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint32_t* __restrict__  shape_src,
    const uint32_t* __restrict__  strides_src,
    const uint32_t* __restrict__  shape_dst,
    const uint32_t* __restrict__  strides_dst,
    const uint32_t ndim_src,
    const uint32_t ndim_dst) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = j * width + i;
        {{ dtype }} tmp = src[index_transform(idx, strides_src, shape_src, ndim_src)]; // In case src == dst
        dst[index_transform(idx, strides_dst, shape_dst, ndim_dst)] = src[index_transform(idx + 2 * width * height, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx + width * height, strides_dst, shape_dst, ndim_dst)] = src[index_transform(idx + width * height, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx + 2 * width * height, strides_dst, shape_dst, ndim_dst)] = tmp; // To fix
    }
}

__global__ void HWC_cvt_color_bgr2rgb_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint32_t* __restrict__  shape_src,
    const uint32_t* __restrict__  strides_src,
    const uint32_t* __restrict__  shape_dst,
    const uint32_t* __restrict__  strides_dst,
    const uint32_t ndim_src,
    const uint32_t ndim_dst) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        uint32_t channels = ndim_src == 2 ? 1 : shape_src[2];
        const uint32_t idx = (j * width + i) * channels;
        {{ dtype }} tmp = src[index_transform(idx, strides_src, shape_src, ndim_src)]; // In case src == dst
        dst[index_transform(idx, strides_dst, shape_dst, ndim_dst)] = src[index_transform(idx + 2, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx + 1, strides_dst, shape_dst, ndim_dst)] = src[index_transform(idx + 1, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx + 2, strides_dst, shape_dst, ndim_dst)] = tmp; // To fix
    }
}

__global__ void CHW_cvt_color_bgr2rgb_{{ dtype }}(
    {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint32_t* __restrict__  shape_src,
    const uint32_t* __restrict__  strides_src,
    const uint32_t* __restrict__  shape_dst,
    const uint32_t* __restrict__  strides_dst,
    const uint32_t ndim_src,
    const uint32_t ndim_dst) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = j * width + i;
        {{ dtype }} tmp = src[index_transform(idx, strides_src, shape_src, ndim_src)]; // In case src == dst
        dst[index_transform(idx, strides_dst, shape_dst, ndim_dst)] = src[ index_transform(idx + 2 * width * height, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx + width * height, strides_dst, shape_dst, ndim_dst)] = src[ index_transform(idx + width * height, strides_src, shape_src, ndim_src)]; // To fix
        dst[index_transform(idx + 2 * width * height, strides_dst, shape_dst, ndim_dst)] = tmp; // To fix
    }
}

