
#include <stdint.h>

__global__ void HWC_cvt_color_rgb2gray_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint8_t channels) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t src_idx = (j * width + i) * channels;
        const uint32_t dst_idx = j * width + i;
        {{ dtype }} r = src[src_idx];
        {{ dtype }} g = src[src_idx + 1];
        {{ dtype }} b = src[src_idx + 2];
        dst[dst_idx] = ({{ dtype }})(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

__global__ void CHW_cvt_color_rgb2gray_{{ dtype }}(
    const {{ dtype  }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint8_t channels) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = j * width + i;
        {{ dtype }} r = src[idx];
        {{ dtype }} g = src[idx + width * height];
        {{ dtype }} b = src[idx + 2 * width * height];
        dst[idx] = ({{ dtype }})(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

__global__ void HWC_cvt_color_bgr2gray_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint8_t channels) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t src_idx = (j * width + i) * channels;
        const uint32_t dst_idx = j * width + i;
        {{ dtype }} r = src[src_idx + 2];
        {{ dtype }} g = src[src_idx + 1];
        {{ dtype }} b = src[src_idx];
        dst[dst_idx] = ({{ dtype }})(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

__global__ void CHW_cvt_color_bgr2gray_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint8_t channels) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = j * width + i;
        {{ dtype }} r = src[idx + 2 * width * height];
        {{ dtype }} g = src[idx + width * height];
        {{ dtype }} b = src[idx];
        dst[idx] = ({{ dtype }})(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

__global__ void HWC_cvt_color_rgb2bgr_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint8_t channels) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = (j * width + i) * channels;
        {{ dtype }} tmp = src[idx]; // In case src == dst
        dst[idx] = src[idx + 2];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = tmp;
    }
}

__global__ void CHW_cvt_color_rgb2bgr_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint8_t channels) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = j * width + i;
        {{ dtype }} tmp = src[idx]; // In case src == dst
        dst[idx] = src[idx + 2 * width * height];
        dst[idx + width * height] = src[idx + width * height];
        dst[idx + 2 * width * height] = tmp;
    }
}

__global__ void HWC_cvt_color_bgr2rgb_{{ dtype }}(
    const {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint8_t channels) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = (j * width + i) * channels;
        {{ dtype }} tmp = src[idx]; // In case src == dst
        dst[idx] = src[idx + 2];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = tmp;
    }
}

__global__ void CHW_cvt_color_bgr2rgb_{{ dtype }}(
    {{ dtype }}* __restrict__ src,
    {{ dtype }}* __restrict__ dst,
    const uint16_t width,
    const uint16_t height,
    const uint8_t channels) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        const uint32_t idx = j * width + i;
        {{ dtype }} tmp = src[idx]; // In case src == dst
        dst[idx] = src[idx + 2 * width * height];
        dst[idx + width * height] = src[idx + width * height];
        dst[idx + 2 * width * height] = tmp;
    }
}

