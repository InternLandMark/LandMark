// MIT License

// Copyright (c) Microsoft Corporation.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#include <torch/extension.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define NUM_BLOCK 256

#define THREAD_NUM 512


template <typename scalar_t>
static __forceinline__ __device__
    scalar_t
    grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners)
{
    if (align_corners)
    {
        // unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + 1) / 2) * (size - 1);
    }
    else
    {
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + 1) * size - 1) / 2;
    }
}

static __forceinline__ __device__ bool within_bounds(int h, int H)
{
    return h >= 0 && h < H;
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__
    scalar_t
    clip_coordinates(scalar_t in, int clip_limit)
{
    return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
}

template <typename scalar_t>
static __forceinline__ __device__
    scalar_t
    safe_downgrade_to_int_range(scalar_t x)
{
    // -100.0 does not have special meaning. This is just to make sure
    // it's not within_bounds_2d or within_bounds_3d, and does not cause
    // undefined behavior. See #35506.
    if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
        return static_cast<scalar_t>(-100.0);
    return x;
}

template <typename scalar_t>
static __forceinline__ __device__
    scalar_t
    compute_coordinates(scalar_t coord, int size,
                        bool padding_mode,
                        bool align_corners)
{
    if (padding_mode)
    { // True for border padding
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    }
    coord = safe_downgrade_to_int_range(coord);
    return coord;
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
static __forceinline__ __device__
    scalar_t
    grid_sampler_compute_source_index(
        scalar_t coord,
        int size,
        bool padding_mode,
        bool align_corners)
{
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    coord = compute_coordinates(coord, size, padding_mode, align_corners);
    return coord;
}
template <typename scalar_t>
__device__ scalar_t gridsample1d_cal(const scalar_t *__restrict__ line, const int line_h_in, const int line_w_in, const float iy, const int channel_offset) {
    int32_t iy_nw = int32_t(iy);
    return ((line[channel_offset * line_h_in * line_w_in + iy_nw * line_w_in] * (float(iy_nw + 1) - iy))) +
           ((line[channel_offset * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in] * (iy - float(iy_nw))));
    // // 1.5d 当能压在线上时,出现此参数配置，该值常态为0
    // int32_t ix_15d = 0;
    // int32_t iy_nw = int32_t(iy);
    // return ((line[channel_offset * line_h_in * line_w_in + iy_nw * line_w_in + ix_15d] * (float(iy_nw + 1) - iy))) +
    //     // ((line[(int)blockIdx.x * line_h_in * line_w_in + iy_nw * line_w_in] * (1.0 - (iy - float(iy_nw))))) +
    //        ((line[channel_offset * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + ix_15d] * (iy - float(iy_nw))));
}

template <typename scalar_t>
__device__ scalar_t gridsample1dot5d_cal(const scalar_t *__restrict__ line, const int line_h_in, const int line_w_in, const int ix, const float iy, const int channel_offset) {
    // 1.5d 当能压在线上时,出现此参数配置，该值常态为0
    int32_t iy_nw = int32_t(iy);
    return ((line[channel_offset * line_h_in * line_w_in + iy_nw * line_w_in + ix] * (float(iy_nw + 1) - iy))) +
           ((line[channel_offset * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + ix] * (iy - float(iy_nw))));
}

// template <typename scalar_t>
// __device__ scalar_t gridsample2dot5d_cal(const scalar_t *__restrict__ plane, const int block_in, const int h_in, const int w_in,
// const float ix, const float iy, const int ib, const int channel_offset) {

//     int b_pos = ib * h_in * w_in;
//     int channel_pos = channel_offset * block_in * h_in * w_in;
//     // 版本1：精度最佳的描述
//     int ix_nw = (int)floorf(ix);
//     int iy_nw = (int)floorf(iy);
//     int ix_ne = ix_nw + 1;
//     int iy_ne = iy_nw;
//     int ix_sw = ix_nw;
//     int iy_sw = iy_nw + 1;
//     int ix_se = ix_nw + 1;
//     int iy_se = iy_nw + 1;
//     // get surfaces to each neighbor:
//     float nw = (ix_se - ix) * (iy_se - iy);
//     float ne = (ix - ix_sw) * (iy_sw - iy);
//     float sw = (ix_ne - ix) * (iy - iy_ne);
//     float se = (ix - ix_nw) * (iy - iy_nw);
//     // calculate bilinear weighted pixel value and set output pixel
//     float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ?
//       plane[channel_pos + b_pos + iy_nw * w_in + ix_nw ] : 0;
//     float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ?
//       plane[channel_pos + b_pos + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] : 0;
//     float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ?
//       plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] : 0;
//     float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ?
//       plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] : 0;
//     return nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
// }

template <typename scalar_t>
__device__ scalar_t gridsample2dot5d_cal(const scalar_t *__restrict__ plane, const int block_in, const int h_in, const int w_in,
const float ix, const float iy, const int ib, const int channel_offset) {

    long b_pos = long(ib) * h_in * w_in;
    long channel_pos = long(channel_offset) * block_in * h_in * w_in;
    // 版本1：精度最佳的描述
    int ix_nw = (int)floorf(ix);
    int iy_nw = (int)floorf(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    // get surfaces to each neighbor:
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);
    // calculate bilinear weighted pixel value and set output pixel
    // if (channel_pos + b_pos + iy_nw * w_in + ix_nw > 2147483640){
    //     printf("%lld\t",channel_pos + b_pos + iy_nw * w_in + ix_nw);
    // }

    float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ?
      plane[channel_pos + b_pos + iy_nw * w_in + ix_nw ] : 0;
    float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ?
      plane[channel_pos + b_pos + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] : 0;
    float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ?
      plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] : 0;
    float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ?
      plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] : 0;
    return nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
}


template <typename scalar_t>
__device__ scalar_t gridsample2d_cal(const scalar_t *__restrict__ plane, const int h_in, const int w_in, const float ix, const float iy, const int channel_offset) {
    // 版本1：精度最佳的描述
    int ix_nw = (int)floorf(ix);
    int iy_nw = (int)floorf(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    // get surfaces to each neighbor:
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);
    // calculate bilinear weighted pixel value and set output pixel
    float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ? plane[channel_offset * h_in * w_in + iy_nw * w_in + ix_nw] : 0;
    float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ? plane[channel_offset * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] : 0;
    float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ? plane[channel_offset * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] : 0;
    float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ? plane[channel_offset * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] : 0;
    return nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
}


// 计算appfeature的基础版本，单次的2d+1d的gridsample并按位相乘
__global__ void fusedkernel1(const float* xyz, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {

    float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
    float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
    float plane_coef_tmp = gridsample2d_cal(plane, h_in, w_in, ix, iy, (int)blockIdx.x);
    iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
    float line_coef_tmp = gridsample1d_cal(line, line_h_in, line_w_in, iy, (int)blockIdx.x);
    sigma[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;

  }
}

// 在基础版本上的优化方案1: 融合
__global__ void fusedkernel2(const float* xyz, const float* plane0, const float* line0, float* sigma0, const int xyz_num, const int h_in0, const int w_in0, const int line_h_in0, const int line_w_in0, const int C,
                                               const float* plane1, const float* line1, float* sigma1,                    const int h_in1, const int w_in1, const int line_h_in1, const int line_w_in1) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {

    float ix0 = ((float(w_in0-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
    float iy0 = ((float(h_in0-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
    float plane_coef_tmp = gridsample2d_cal(plane0, h_in0, w_in0, ix0, iy0, (int)blockIdx.x);
    iy0 = ((float(line_h_in0-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
    float line_coef_tmp = gridsample1d_cal(line0, line_h_in0, line_w_in0, iy0, (int)blockIdx.x);
    sigma0[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;

    float ix1 = ((float(w_in1-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
    float iy1 = ((float(h_in1-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
    plane_coef_tmp = gridsample2d_cal(plane1, h_in1, w_in1, ix1, iy1, (int)blockIdx.x);
    iy1 = ((float(line_h_in1-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
    line_coef_tmp = gridsample1d_cal(line1, line_h_in1, line_w_in1, iy1, (int)blockIdx.x);
    sigma1[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;
  }
}

// 在基础版本上的优化方案2: 串行化
__global__ void fusedkernel_serial(const float* xyz, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C, const int serial) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
    for (int cc = 0; cc < serial; cc++) {
      float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
      float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
      int channel_offset = (blockIdx.x * serial + cc);
      float plane_coef_tmp = gridsample2d_cal(plane, h_in, w_in, ix, iy, channel_offset);
      iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
      float line_coef_tmp = gridsample1d_cal(line, line_h_in, line_w_in, iy, channel_offset);
      sigma[channel_offset * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;
    }
  }
}

// 在基础版本上的优化方案2: 串行化，并支持xyzb
__global__ void fusedkernel_serial_xyzb(const float* xyzb, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C, const int serial) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
    for (int cc = 0; cc < serial; cc++) {
      float ix = ((float(w_in-1) * (xyzb[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 4 + 0] + 1.0)) / 2.0);
      float iy = ((float(h_in-1) * (xyzb[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 4 + 1] + 1.0)) / 2.0);
      int channel_offset = (blockIdx.x * serial + cc);
      float plane_coef_tmp = gridsample2d_cal(plane, h_in, w_in, ix, iy, channel_offset);

      // ix = ((float(line_w_in-1) * (xyzb[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 4 + 3] + 1.0)) / 2.0) ;
      ix = ((float(line_w_in-1) * (xyzb[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 4 + 3] + 1.0)) / 2.0) + 0.0001;

      iy = ((float(line_h_in-1) * (xyzb[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 4 + 2] + 1.0)) / 2.0);

      // float line_coef_tmp = gridsample2d_cal(line, line_h_in, line_w_in, ix, iy, channel_offset);
      float line_coef_tmp = gridsample1dot5d_cal(line, line_h_in, line_w_in, int32_t(ix), iy, channel_offset);
      sigma[channel_offset * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;
    }
  }
}

// 与xyzb版本不同点在于，xyz矩阵与b矩阵分离
__global__ void fusedkernel_serial_xyz_b(const float* xyz, const int* b, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C, const int serial) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
    for (int cc = 0; cc < serial; cc++) {
      float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
      float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
      int channel_offset = (blockIdx.x * serial + cc);
      float plane_coef_tmp = gridsample2d_cal(plane, h_in, w_in, ix, iy, channel_offset);

      int ib = b[((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x];
      iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
      float line_coef_tmp = gridsample1dot5d_cal(line, line_h_in, line_w_in, ib, iy, channel_offset);
      sigma[channel_offset * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;
    }
  }
}
// 与xyzb版本不同点在于，xyz矩阵与b矩阵分离，且xyb
__global__ void fusedkernel_serial_xyb_bz(const float* xyz, const int* b, const float* plane, const float* line, float* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int C, const int serial) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {

    int ib = b[((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x];

    for (int cc = 0; cc < serial; cc++) {
      float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
      float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
      int channel_offset = (blockIdx.x * serial + cc);
      // float plane_coef_tmp = gridsample2d_cal(plane, h_in, w_in, ix, iy, channel_offset);
      float plane_coef_tmp = gridsample2dot5d_cal(plane, block_in, h_in, w_in, ix, iy, ib, cc);

      iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
      float line_coef_tmp = gridsample1dot5d_cal(line, line_h_in, line_w_in, ib, iy, channel_offset);
      sigma[channel_offset * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;
    }
  }
}

template <typename scalar_t>
__device__ float gridsample1dot5d_halfcal(const scalar_t *__restrict__ line, const int line_h_in, const int line_w_in, const int ix, const float iy, const int channel_offset) {
    // 1.5d 当能压在线上时,出现此参数配置，该值常态为0
    int32_t iy_nw = int32_t(iy);
    return (( 	__half2float(line[channel_offset * line_h_in * line_w_in + iy_nw * line_w_in + ix]) * (float(iy_nw + 1) - iy))) +
           ((	__half2float(line[channel_offset * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + ix]) * (iy - float(iy_nw))));
}

template <typename scalar_t>
__device__ float gridsample2dot5d_halfblockcal(const scalar_t *__restrict__ plane_block, const int block_in, const int h_in, const int w_in,
const float ix, const float iy, const int ib, const int channel_offset) {

    int b_pos = ib * h_in * w_in;  //最大值为 23x2061x4001
    // long channel_pos = channel_offset * block_in * h_in * w_in; //最大值为47*24x2061x4001 有越界的风险
    // 版本1：精度最佳的描述
    int ix_nw = (int)floorf(ix);
    int iy_nw = (int)floorf(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    // get surfaces to each neighbor:
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);
    // calculate bilinear weighted pixel value and set output pixel
    float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ?
      __half2float(plane_block[b_pos + iy_nw * w_in + ix_nw ]) : 0;
    float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ?
      __half2float(plane_block[b_pos + iy_nw * w_in + min((ix_nw + 1), (w_in-1))]) : 0;
    float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ?
      __half2float(plane_block[b_pos + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw]) : 0;
    float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ?
      __half2float(plane_block[b_pos + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))]) : 0;
    return nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
}

__global__ void fusedkernel_serial_xyb_bz_half(const float* xyz, const int* b, const half* plane, const half* line, half* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int C, const int serial) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {

    int ib = b[((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x];

    for (int cc = 0; cc < serial; cc++) {
      float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
      float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
      int channel_offset = (blockIdx.x * serial + cc);
      // float plane_coef_tmp = gridsample2d_cal(plane, h_in, w_in, ix, iy, channel_offset);
      // float plane_coef_tmp = gridsample2dot5d_halfcal(plane, block_in, h_in, w_in, ix, iy, ib, cc);

      long channel_pos = long(cc) * block_in * h_in * w_in;
      float plane_coef_tmp = gridsample2dot5d_halfblockcal(&plane[channel_pos], block_in, h_in, w_in, ix, iy, ib, channel_offset);

      iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
      float line_coef_tmp = gridsample1dot5d_halfcal(line, line_h_in, line_w_in, ib, iy, channel_offset);
      sigma[channel_offset * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = __float2half(plane_coef_tmp * line_coef_tmp);
    }
  }
}

int compute_grid_sample_and_ewproduct_serial_xyb_bz_half_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);
    // plane shape (N,C,h_in,w_in)  [1, 48, 2744, 2744]
    // plane shape (N,C,B,h_in,w_in)  [1, 48, 8, 2744, 2744]
    const int n = plane.size(0); // n == 1
    const int c = plane.size(1); // c == 48
    const int block_in = plane.size(2); // block_in == 8
    const int h_in = plane.size(2+1); // h_in == 2047
    const int w_in = plane.size(3+1); // w_in == 2047
    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    assert( c % serial == 0);
    assert(block_in==line_w_in);

    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c / serial, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_xyb_bz_cuda", ([&] {
    fusedkernel_serial_xyb_bz_half<<<grid, block>>>(
        xyz.data<float>(),
        b.data<int>(),
        reinterpret_cast<half*> (plane.data_ptr<torch::Half>()),
        reinterpret_cast<half*> (line.data_ptr<torch::Half>()),
        reinterpret_cast<half*> (sigma.data_ptr<torch::Half>()),
        xyz_num,
        block_in,
        h_in,
        w_in,
        line_h_in,
        line_w_in,
        c,
        serial);
    }));
    return 0;
}

// 在xybbz的基础上, 进行channel循环的下放, 提高了25%的速度, 并把输出转为half
__global__ void fusedkernel_serial_xyb_bz_outputhalf(const float* xyz, const int* b, const float* plane, const float* line, half* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int C, const int serial) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {

    int ib = b[((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x];


      float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
      float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
      float iz = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
      int iz_nw = int(iz);

      // float plane_coef_tmp = gridsample2d_cal(plane, h_in, w_in, ix, iy, channel_offset);
      // printf("here!");

      long b_pos = long(ib) * h_in * w_in;

      // 版本1：精度最佳的描述
      int ix_nw = (int)floorf(ix);
      int iy_nw = (int)floorf(iy);
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;
      // get surfaces to each neighbor:
      float nw = (ix_se - ix) * (iy_se - iy);
      float ne = (ix - ix_sw) * (iy_sw - iy);
      float sw = (ix_ne - ix) * (iy - iy_ne);
      float se = (ix - ix_nw) * (iy - iy_nw);
      // calculate bilinear weighted pixel value and set output pixel
      // if (channel_pos + b_pos + iy_nw * w_in + ix_nw > 2147483640){
      //     printf("%lld\t",channel_pos + b_pos + iy_nw * w_in + ix_nw);
      // }
      for (int cc = 0; cc < serial; cc++) {
        int channel_offset = (blockIdx.x * serial + cc);
        long channel_pos = long(channel_offset) * block_in * h_in * w_in;

        float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ?
          plane[channel_pos + b_pos + iy_nw * w_in + ix_nw ] : 0;
        float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ?
          plane[channel_pos + b_pos + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] : 0;
        float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ?
          plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] : 0;
        float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ?
          plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] : 0;
        float plane_coef_tmp =  nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;


        float line_coef_tmp =((line[channel_offset * line_h_in * line_w_in + iz_nw * line_w_in + ib] * (float(iz_nw + 1) - iz))) +
            ((line[channel_offset * line_h_in * line_w_in + min((iz_nw + 1), (line_h_in-1)) * line_w_in + ib] * (iz - float(iz_nw))));

        sigma[channel_offset * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = __float2half(plane_coef_tmp * line_coef_tmp);
    }
  }
}

int compute_grid_sample_and_ewproduct_serial_xyb_bz_outhalf_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);
    // plane shape (N,C,h_in,w_in)  [1, 48, 2744, 2744]
    // plane shape (N,C,B,h_in,w_in)  [1, 48, 8, 2744, 2744]
    const int n = plane.size(0); // n == 1
    const int c = plane.size(1); // c == 48
    const int block_in = plane.size(2); // block_in == 8
    const int h_in = plane.size(2+1); // h_in == 2047
    const int w_in = plane.size(3+1); // w_in == 2047
    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    assert( c % serial == 0);
    assert(block_in==line_w_in);

    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c / serial, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_xyb_bz_cuda", ([&] {
    fusedkernel_serial_xyb_bz_outputhalf<<<grid, block>>>(
        xyz.data<float>(),
        b.data<int>(),
        plane.data<float>(),
        line.data<float>(),
        reinterpret_cast<half*> (sigma.data_ptr<torch::Half>()),
        xyz_num,
        block_in,
        h_in,
        w_in,
        line_h_in,
        line_w_in,
        c,
        serial);
    }));
    return 0;
}

// // 在优化方案2基础上的优化方案3: 向量化  效果不佳,暂无使用
// __global__ void fusedkernel_serial_vec(float* xyz, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C, const int serial) {
//   // printf("threadIdx.x: %d \t min: %d \t  is_true?%d\n", threadIdx.x, (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1), (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))));
//   if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
//     for (int cc = 0; cc < serial; cc++) {
//       float inputs3[3];
//       reinterpret_cast<float3*>(inputs3)[0] = reinterpret_cast<float3*>(xyz + (blockIdx.y * NUM_BLOCK+ threadIdx.x) * 3)[0];
//       float ix;
//       float iy;
//       float iz;

//       int32_t ix_nw;
//       int32_t iy_nw;

//       ix = ((float(w_in-1) * (inputs3[0] + 1.0)) / 2.0);
//       iy = ((float(h_in-1) * (inputs3[1] + 1.0)) / 2.0);
//       iz = ((float(line_h_in-1) * (inputs3[2] + 1.0)) / 2.0);

//       // 向下取整
//       ix_nw = int32_t(ix);
//       iy_nw = int32_t(iy);
//       int channel_offset;
//       channel_offset = (blockIdx.x * serial + cc);
//       float plane_coef_tmp;
//       plane_coef_tmp = (((((plane[channel_offset * h_in * w_in + iy_nw * w_in + ix_nw] * (1.0 - (iy - float(iy_nw)))) * (1.0  - (ix - float(ix_nw)))) +
//                           ((plane[channel_offset * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] * (iy - float(iy_nw))) * (1.0  - (ix - float(ix_nw))))) +
//                           ((plane[channel_offset * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] * (1.0  - (iy - float(iy_nw)))) * (ix - float(ix_nw)))) +
//                           ((plane[channel_offset * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] * (iy - float(iy_nw))) * (ix - float(ix_nw))));
//       iy_nw = int32_t(iz);
//       float line_coef_tmp;
//       line_coef_tmp = ((line[channel_offset * line_h_in * line_w_in + iy_nw * line_w_in] * (1.0 - (iz - float(iy_nw))))) +
//                       ((line[channel_offset * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in] * (iz - float(iy_nw))));
//       sigma[channel_offset * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;
//     }
//   }
// }

//  计算density的版本，单次计算两个的gridsample按位相乘后求和, 按xyz处理
__global__ void fusedkernel_sum(const float* xyz, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
    float sum;
    sum = 0.0;
    for (int cc = 0; cc < C; cc++) {
      float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
      float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
      float plane_coef_tmp = gridsample2d_cal(plane, h_in, w_in, ix, iy, cc);
      iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
      float line_coef_tmp = gridsample1d_cal(line, line_h_in, line_w_in, iy, cc);
      sum += plane_coef_tmp * line_coef_tmp;
    }
    sigma[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = sum;
  }
}

// 计算density的版本，单次计算两个的gridsample按位相乘后求和, 按xyzb处理
// 注意：唯有该函数增加了边界判断，对非法xyzb输入进行判断
__global__ void fusedkernel_sum_xyzb(const float* xyzb, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C, bool judgeOverflow) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
    float sum;
    sum = 0.0;
    for (int cc = 0; cc < C; cc++) {
      float ix, iy;
      float result_plane, result_line;

      if (judgeOverflow && (abs(xyzb[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 4 + 0]) > 1.0 || abs(xyzb[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 4 + 1]) > 1.0)){
        result_plane = 0.0;
      }else{
        ix = ((float(w_in-1) * (xyzb[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 4 + 0] + 1.0)) / 2.0);
        iy = ((float(h_in-1) * (xyzb[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 4 + 1] + 1.0)) / 2.0);
        result_plane = gridsample2d_cal(plane, h_in, w_in, ix, iy, cc);
      }

      if (judgeOverflow && (abs(xyzb[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 4 + 2]) > 1.0 || abs(xyzb[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 4 + 3]) > 1.0)){
        result_line = 0.0;
      }else{
        ix = ((float(line_w_in-1) * (xyzb[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 4 + 3] + 1.0)) / 2.0);
        iy = ((float(line_h_in-1) * (xyzb[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 4 + 2] + 1.0)) / 2.0);
        result_line = gridsample2d_cal(line, line_h_in, line_w_in, ix, iy, cc);

      }
      sum += result_plane * result_line;
    }
    sigma[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = sum;
  }
}

// 计算density的版本，单次计算两个的gridsample按位相乘后求和, 按xyzb处理
// 与xyzb版本不同点在于，xyz矩阵与b矩阵分离
__global__ void fusedkernel_sum_xyz_b(const float* xyz, const int* b, const float* plane, const float* line, float* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int C, bool judgeOverflow) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
    float sum;
    sum = 0.0;

    unsigned int xyzlen = 3; //4
    int ib = b[((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x];

    for (int cc = 0; cc < C; cc++) {
      float ix, iy;
      float result_plane, result_line;

      if (judgeOverflow && (abs(xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 0]) > 1.0 || abs(xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 1]) > 1.0)){
        result_plane = 0.0;
      }else{
        ix = ((float(w_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 0] + 1.0)) / 2.0);
        iy = ((float(h_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 1] + 1.0)) / 2.0);
        // result_plane = gridsample2d_cal(plane, h_in, w_in, ix, iy, cc);
        result_plane = gridsample2dot5d_cal(plane, block_in, h_in, w_in, ix, iy, ib, cc);
      }

      if (judgeOverflow && (abs(xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 2]) > 1.0 || abs(xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 3]) > 1.0)){
        result_line = 0.0;
      }else{
        //ix = ((float(line_w_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 3] + 1.0)) / 2.0);
        iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 2] + 1.0)) / 2.0);
        result_line = gridsample1dot5d_cal(line, line_h_in, line_w_in, ib, iy, cc);
      }
      sum += result_plane * result_line;
    }
    sigma[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = sum;
  }
}
// __global__ void fusedkernel_sum_xyz_b(const float* xyz, const int* b, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C, bool judgeOverflow) {
//   if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
//     float sum;
//     sum = 0.0;

//     unsigned int xyzlen = 3; //4
//     for (int cc = 0; cc < C; cc++) {
//       float ix, iy;
//       float result_plane, result_line;

//       if (judgeOverflow && (abs(xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 0]) > 1.0 || abs(xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 1]) > 1.0)){
//         result_plane = 0.0;
//       }else{
//         ix = ((float(w_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 0] + 1.0)) / 2.0);
//         iy = ((float(h_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 1] + 1.0)) / 2.0);
//         result_plane = gridsample2d_cal(plane, h_in, w_in, ix, iy, cc);
//       }

//       if (judgeOverflow && (abs(xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 2]) > 1.0 || abs(xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 3]) > 1.0)){
//         result_line = 0.0;
//       }else{
//         int ib = b[((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x];
//         //ix = ((float(line_w_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 3] + 1.0)) / 2.0);
//         iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * xyzlen + 2] + 1.0)) / 2.0);
//         result_line = gridsample1dot5d_cal(line, line_h_in, line_w_in, ib, iy, cc);
//       }
//       sum += result_plane * result_line;
//     }
//     sigma[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = sum;
//   }
// }


__global__ void kernel1d(const float* xyz, const float* line, float* line_coef,
const int xyz_num,  const int line_h_in, const int line_w_in) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {

    float y = xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2];
    float iy = ((float(line_h_in-1) * (y + 1.0)) / 2.0);
    iy = safe_downgrade_to_int_range(iy);
    line_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = gridsample1d_cal(line, line_h_in, line_w_in, iy, (int)blockIdx.x);
  }
}

template <typename scalar_t>
__global__ void kernel2d(const scalar_t *__restrict__ xyz,
const scalar_t *__restrict__ plane,  scalar_t *__restrict__ plane_coef, /*const float* line, float* line_coef, */
const int xyz_num, const int h_in, const int w_in  /*, const int line_h_in, const int line_w_in*/) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
    scalar_t x = xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0];
    scalar_t y = xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1];
    // ix = ((scalar_t(w_in-1) * (x + 1.0)) / 2.0);
    // iy = ((scalar_t(h_in-1) * (y + 1.0)) / 2.0);
    float ix = grid_sampler_compute_source_index(x, h_in, false, true);
    float iy = grid_sampler_compute_source_index(y, w_in, false, true);
    ix = safe_downgrade_to_int_range(ix);
    iy = safe_downgrade_to_int_range(iy);
    plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = gridsample2d_cal(plane, h_in, w_in, ix, iy, (int)blockIdx.x);

    // // 版本1：精度最佳的描述
    // int ix_nw = (int)floorf(ix);
    // int iy_nw = (int)floorf(iy);
    // int ix_ne = ix_nw + 1;
    // int iy_ne = iy_nw;
    // int ix_sw = ix_nw;
    // int iy_sw = iy_nw + 1;
    // int ix_se = ix_nw + 1;
    // int iy_se = iy_nw + 1;
    // // get surfaces to each neighbor:
    // float nw = (ix_se - ix) * (iy_se - iy);
    // float ne = (ix - ix_sw) * (iy_sw - iy);
    // float sw = (ix_ne - ix) * (iy - iy_ne);
    // float se = (ix - ix_nw) * (iy - iy_nw);
    // // calculate bilinear weighted pixel value and set output pixel
    // float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ? plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + ix_nw] : 0;
    // float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ? plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] : 0;
    // float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ? plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] : 0;
    // float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ? plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] : 0;
    // float out_val = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
    // plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = out_val;

    // 版本2: 使用内部函数——精度有所下降
    // int ix_nw = (int)floorf(ix);
    // int iy_nw = (int)floorf(iy);
    // int ix_ne = __fadd_rn(ix_nw, 1);
    // int iy_ne = iy_nw;
    // int ix_sw = ix_nw;
    // int iy_sw = __fadd_rn(iy_nw,1);
    // int ix_se = __fadd_rn(ix_nw,1);
    // int iy_se = __fadd_rn(iy_nw,1);
    // float nw = __fmul_rn(__fsub_rn(ix_se, ix), __fsub_rn(iy_se, iy));
    // float ne = __fmul_rn(__fsub_rn(ix, ix_sw), __fsub_rn(iy_sw, iy));
    // float sw = __fmul_rn(__fsub_rn(ix_ne, ix), __fsub_rn(iy, iy_ne));
    // float se = __fmul_rn(__fsub_rn(ix, ix_nw), __fsub_rn(iy, iy_nw));
    // float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ? plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + ix_nw] : 0;
    // float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ? plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] : 0;
    // float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ? plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] : 0;
    // float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ? plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] : 0;
    // float out_val = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(nw_val, nw), __fmul_rn(ne_val, ne)), __fmul_rn(sw_val, sw)), __fmul_rn(se_val, se));
    // plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = out_val;

    // 版本3:较杂糅
    // float nw = (scalar_t(ix_nw + 1) - ix) * (scalar_t(iy_nw + 1) - iy);
    // float ne = (ix - scalar_t(ix_nw)) * (scalar_t(iy_nw + 1) - iy);
    // float sw = (scalar_t(ix_nw + 1) - ix) * (iy - scalar_t(iy_nw));
    // float se = (ix - scalar_t(ix_nw)) * (iy - scalar_t(iy_nw));
    // float nw = ((ix_nw + 1) - ix) * ((iy_nw + 1) - iy);
    // float ne = (ix - ix_nw) * ((iy_nw + 1) - iy);
    // float sw = ((ix_nw + 1) - ix) * (iy - iy_nw);
    // float se = (ix - ix_nw) * (iy - iy_nw);
    // plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] =
    // ((plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + ix_nw] * nw) +
    // (plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] * ne) +
    // (plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] * sw) +
    // (plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] * se));

    // 版本4:最杂糅
    // const int ix_nw = ::floor(ix);
    // const int iy_nw = ::floor(iy);
    // // plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = static_cast<scalar_t>(0);
    // plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] =
    // (((((plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + ix_nw] * (1.0 - (iy - scalar_t(iy_nw)))) * (1.0  - (ix - scalar_t(ix_nw)))) +
    // ((plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] * (iy - scalar_t(iy_nw))) * (1.0  - (ix - scalar_t(ix_nw))))) +
    // ((plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] * (1.0  - (iy - scalar_t(iy_nw)))) * (ix - scalar_t(ix_nw)))) +
    // ((plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] * (iy - scalar_t(iy_nw))) * (ix - scalar_t(ix_nw))));
  }
}

// 计算appfeature的基础版本，单次的2d+1d的gridsample并按位相乘
int compute_grid_sample_and_ewproduct_cuda(torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);
    // plane shape (N,C,h_in,w_in)  [1, 48, 2744, 2744]
    const int n = plane.size(0); // n == 1
    // assert(n==1);
    const int c = plane.size(1); // c == 48
    const int h_in = plane.size(2); // h_in == 2047
    const int w_in = plane.size(3); // w_in == 2047
    // line shape (N,C,line_h_in,line_w_in) [1, 48, 228, 1]
    // assert(line.size(0) == n);
    // assert(line.size(1) == c);
    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1

    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_cuda", ([&] {
        fusedkernel1<<<grid, block>>>(
            xyz.data<float>(),
            plane.data<float>(),
            line.data<float>(),
            sigma.data<float>(),
            xyz_num,
            h_in,
            w_in,
            line_h_in,
            line_w_in,
            c);
        }));

    return 0;
}
// 加速方案1：合并两个基础版本
int compute_grid_sample_and_ewproduct_cuda2(torch::Tensor xyz, torch::Tensor plane0, torch::Tensor line0, torch::Tensor sigma0,
                                                               torch::Tensor plane1, torch::Tensor line1, torch::Tensor sigma1)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);

    // plane shape (N,C,h_in,w_in)  [1, 48, 2744, 2744]
    const int n = plane0.size(0); // n == 1
    const int c = plane0.size(1); // c == 48
    const int h_in0 = plane0.size(2); // h_in == 2047
    const int w_in0 = plane0.size(3); // w_in == 2047

    // line shape (N,C,line_h_in,line_w_in) [1, 48, 228, 1]
    const int line_h_in0 = line0.size(2); // line_h_in == 255
    const int line_w_in0 = line0.size(3); // line_w_in == 1

    const int h_in1 = plane1.size(2); // h_in == 2047
    const int w_in1 = plane1.size(3); // w_in == 2047

    // line shape (N,C,line_h_in,line_w_in) [1, 48, 228, 1]
    const int line_h_in1 = line1.size(2); // line_h_in == 255
    const int line_w_in1 = line1.size(3); // line_w_in == 1

    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);

    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_cuda2", ([&] {
        fusedkernel2<<<grid, block>>>(
            xyz.data<float>(),
            plane0.data<float>(),
            line0.data<float>(),
            sigma0.data<float>(),
            xyz_num,
            h_in0,
            w_in0,
            line_h_in0,
            line_w_in0,
            c,
            plane1.data<float>(),
            line1.data<float>(),
            sigma1.data<float>(),
            h_in1,
            w_in1,
            line_h_in1,
            line_w_in1);
        }));

    return 0;
}
// 加速方案2：对基础版本做串行化
int compute_grid_sample_and_ewproduct_serial_cuda(torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);
    // plane shape (N,C,h_in,w_in)  [1, 48, 2744, 2744]
    const int n = plane.size(0); // n == 1
    const int c = plane.size(1); // c == 48
    const int h_in = plane.size(2); // h_in == 2047
    const int w_in = plane.size(3); // w_in == 2047
    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    assert( c % serial == 0);
    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c / serial, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);
    if (xyz.size(1) == 3){
      AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_cuda", ([&] {
          fusedkernel_serial<<<grid, block>>>(
              xyz.data<float>(),
              plane.data<float>(),
              line.data<float>(),
              sigma.data<float>(),
              xyz_num,
              h_in,
              w_in,
              line_h_in,
              line_w_in,
              c,
              serial);
          }));
    }else if (xyz.size(1) == 4){
      AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_cuda", ([&] {
      fusedkernel_serial_xyzb<<<grid, block>>>(
          xyz.data<float>(),
          plane.data<float>(),
          line.data<float>(),
          sigma.data<float>(),
          xyz_num,
          h_in,
          w_in,
          line_h_in,
          line_w_in,
          c,
          serial);
      }));
    }
    return 0;
}


int compute_grid_sample_and_ewproduct_serial_xyz_b_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);
    // plane shape (N,C,h_in,w_in)  [1, 48, 2744, 2744]
    const int n = plane.size(0); // n == 1
    const int c = plane.size(1); // c == 48
    const int h_in = plane.size(2); // h_in == 2047
    const int w_in = plane.size(3); // w_in == 2047
    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    assert( c % serial == 0);
    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c / serial, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_xyz_b_cuda", ([&] {
    fusedkernel_serial_xyz_b<<<grid, block>>>(
        xyz.data<float>(),
        b.data<int>(),
        plane.data<float>(),
        line.data<float>(),
        sigma.data<float>(),
        xyz_num,
        h_in,
        w_in,
        line_h_in,
        line_w_in,
        c,
        serial);
    }));
    return 0;
}

int compute_grid_sample_and_ewproduct_serial_xyb_bz_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);
    // plane shape (N,C,h_in,w_in)  [1, 48, 2744, 2744]
    const int n = plane.size(0); // n == 1
    const int c = plane.size(1); // c == 48
    const int block_in = plane.size(2); // block_in == 8
    const int h_in = plane.size(2+1); // h_in == 2047
    const int w_in = plane.size(3+1); // w_in == 2047
    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    assert( c % serial == 0);
    assert(block_in==line_w_in);

    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c / serial, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_xyb_bz_cuda", ([&] {
    fusedkernel_serial_xyb_bz<<<grid, block>>>(
        xyz.data<float>(),
        b.data<int>(),
        plane.data<float>(),
        line.data<float>(),
        sigma.data<float>(),
        xyz_num,
        block_in,
        h_in,
        w_in,
        line_h_in,
        line_w_in,
        c,
        serial);
    }));
    return 0;
}

//  计算density的版本，单次计算两个的gridsample按位相乘后求和,
//  若输入xyz则进行2d+1d, 若输入tensor为xyzb则进行2d+2d
int compute_grid_sample_and_sum_cuda(
    torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma)
{
    const int xyz_num = xyz.size(0);

    // plane shape (N,C,h_in,w_in)
    const int n = plane.size(0); // n == 1
    // assert(n==1);
    const int c = plane.size(1); // c == 16
    const int h_in = plane.size(2); // h_in == 2047
    const int w_in = plane.size(3); // w_in == 2047

    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1

    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(int((xyz_num - 1) / NUM_BLOCK) + 1, 1, 1);
    if (xyz.size(1) == 3){
      AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_sum_cuda", ([&] {
              fusedkernel_sum<<<grid, block>>>(
                  xyz.data<float>(),
                  plane.data<float>(),
                  line.data<float>(),
                  sigma.data<float>(),
                  xyz_num,
                  h_in,
                  w_in,
                  line_h_in,
                  line_w_in,
                  c);
              }));
    }else if (xyz.size(1) == 4){
      AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_sum_cuda", ([&] {
              fusedkernel_sum_xyzb<<<grid, block>>>(
                  xyz.data<float>(),
                  plane.data<float>(),
                  line.data<float>(),
                  sigma.data<float>(),
                  xyz_num,
                  h_in,
                  w_in,
                  line_h_in,
                  line_w_in,
                  c,
                  false);
              }));
    }
    return 0;
}

int compute_grid_sample_and_sum_xyz_b_cuda(
  torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma)
{
    const int xyz_num = xyz.size(0);

    // plane shape (N,C,h_in,w_in)
    const int n = plane.size(0); // n == 1
    const int c = plane.size(1); // c == 16
    const int block_in = plane.size(2); // block_in == 8
    const int h_in = plane.size(2 + 1); // h_in == 2047
    const int w_in = plane.size(3 + 1); // w_in == 2047

    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    assert(block_in==line_w_in);

    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(int((xyz_num - 1) / NUM_BLOCK) + 1, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_sum_xyz_b_cuda", ([&] {
    fusedkernel_sum_xyz_b<<<grid, block>>>(
        xyz.data<float>(),
        b.data<int>(),
        plane.data<float>(),
        line.data<float>(),
        sigma.data<float>(),
        xyz_num,
        block_in,
        h_in,
        w_in,
        line_h_in,
        line_w_in,
        c,
        false);
    }));
    return 0;
}

int GridSample1D_cuda(torch::Tensor xyz, torch::Tensor plane, torch::Tensor sigma)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);

    // plane shape (N,C,h_in,w_in)  [1, 48, 2744, 2744]
    const int n = plane.size(0); // n == 1
    // assert(n==1);
    const int c = plane.size(1); // c == 48
    const int h_in = plane.size(2); // h_in == 2047

    const int w_in = plane.size(3); // w_in == 2047

    // return {plane_coef, line_coef};
    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_line_test", ([&] {
        kernel1d<<<grid, block>>>(
            xyz.data<float>(),
            plane.data<float>(),
            sigma.data<float>(),
            xyz_num,
            h_in,
            w_in);
        }));

    return 0;
}
int GridSample2D_cuda(torch::Tensor xyz, torch::Tensor plane, torch::Tensor sigma)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);

    // plane shape (N,C,h_in,w_in)  [1, 48, 2744, 2744]
    const int n = plane.size(0); // n == 1
    const int c = plane.size(1); // c == 48
    const int h_in = plane.size(2); // h_in == 2047
    const int w_in = plane.size(3); // w_in == 2047
    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_samplet_test", ([&] {
        kernel2d<scalar_t><<<grid, block>>>(
            xyz.data<scalar_t>(),
            plane.data<scalar_t>(),
            sigma.data<scalar_t>(),
            xyz_num,
            h_in,
            w_in);
        }));
    return 0;
}
