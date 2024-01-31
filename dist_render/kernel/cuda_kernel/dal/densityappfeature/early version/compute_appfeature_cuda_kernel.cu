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
#include <typeinfo>
#include <vector>

#define NUM_BLOCK 256
#define NUM_BLOCK_64 16

#define THREAD_NUM 512

#define IDX2C(i,j,ld)  ((j)*(ld)+(i))
#define IDX3C(i,j,k,rown,coln)  ((k)*(rown)*(coln)+(j)*(rown)+(i))
#define IDX4C(i,j,k,n,rown,coln,depthn)  ((n)*(rown)*(coln)*(depthn)+(k)*(rown)*(coln)+(j)*(rown)+(i))

__forceinline__ __device__ bool inrange(int x, int d) { return x >= 0 && x < d; }

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
__device__ scalar_t gridsample1dot5d_cal(const scalar_t *__restrict__ line, const int line_h_in, const int line_w_in, const int ib, const float iz, const int channel_offset) {
    // 1.5d 当能压在线上时,出现此参数配置，该值常态为0
    int32_t iz_nw = int32_t(iz);
    return ((line[channel_offset * line_h_in * line_w_in + iz_nw * line_w_in + ib] * (float(iz_nw + 1) - iz))) +
           ((line[channel_offset * line_h_in * line_w_in + min((iz_nw + 1), (line_h_in-1)) * line_w_in + ib] * (iz - float(iz_nw))));
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
    // calculate bilinear weighted pixel value and set output pixel    block(c, n_num)     n_num * c * 4
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

// 与xyzb版本不同点在于，xyz矩阵与b矩阵分离，且xyb
__global__ void fusedkernel_serial_xyb_bz_nbhwc(const float* xyz, const int* b, float* plane, const float* line, float* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int C, const int serial) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {

    int float4_len = 4;
    int point_num = 4;
    int shared_per_thread = float4_len * point_num;
    __shared__ float plane_channel[10000];
    float4* plane4 = reinterpret_cast<float4*>(plane);
    int ib = b[((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x];
    // long b_pos = long(ib) * h_in * w_in * C;

    float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
    float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
    int ix_nw = (int)floorf(ix);
    int iy_nw = (int)floorf(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);

    float iz = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
    int32_t iz_nw = int32_t(iz);

    int C_outer = C / float4_len;
    int base_index = 0;
    for (int cc_outer = 0; cc_outer < C_outer; cc_outer++) {
      base_index = (int)threadIdx.x * point_num;
        ((float4*)plane_channel)[base_index + 0] =
        plane4[(long(ib) * h_in * w_in + iy_nw * w_in + ix_nw) * C_outer + cc_outer];
        ((float4*)plane_channel)[base_index + 1] =
        plane4[(long(ib) * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))) * C_outer + cc_outer];
        ((float4*)plane_channel)[base_index + 2] =
        plane4[(long(ib) * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw) * C_outer + cc_outer];
        ((float4*)plane_channel)[base_index + 3] =
        plane4[(long(ib) * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))) * C_outer + cc_outer];


      for(int cc_inner = 0; cc_inner < float4_len; cc_inner++){

        int channel_offset = (blockIdx.x * serial + cc_outer * float4_len + cc_inner);
        // long channel_pos = long(channel_offset) * block_in * h_in * w_in;
        base_index = (int)threadIdx.x * shared_per_thread + cc_inner;
        float nw_val  = plane_channel[ base_index + 0 * float4_len ];
        float ne_val  = plane_channel[ base_index + 1 * float4_len ];
        float sw_val  = plane_channel[ base_index + 2 * float4_len ];
        float se_val  = plane_channel[ base_index + 3 * float4_len ];

        float plane_coef_tmp = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;


        float line_coef_tmp = ((line[channel_offset * line_h_in * line_w_in + iz_nw * line_w_in + ib] * (float(iz_nw + 1) - iz))) +
              ((line[channel_offset * line_h_in * line_w_in + min((iz_nw + 1), (line_h_in-1)) * line_w_in + ib] * (iz - float(iz_nw))));

        sigma[channel_offset * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;
      }
    }
  }
}
// 与xyzb版本不同点在于，xyz矩阵与b矩阵分离，且xyb
__global__ void fusedkernel_serial_xyb_bz_nbhwc2(const float* xyz, const int* b, float* plane, const float* line, float* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int C, const int nanobatch, const int plane_idx, const int plane_in) {
  __shared__ float sample_grid_shared[8 * 8 * 10];  // 一个block上有8个job组，每组有8个有效点数，每个有效点数需要10个float空间存数

  float *sample_grid_thisjob = sample_grid_shared + threadIdx.y * 80;

  int job_id = blockIdx.x * blockDim.y + threadIdx.y;  // blockDim.y == job_num_per_block
  int point_pos_base = job_id * nanobatch;
  // 一个job组中的一些变量
  if(threadIdx.x < nanobatch * 3){
    int nanoit   = threadIdx.x / 3;
    int read_idx = threadIdx.x % 3;
    int point_pos = point_pos_base + nanoit;
    if (point_pos < xyz_num) {
      switch (read_idx) {
        case 0:
            sample_grid_thisjob[threadIdx.x] = ((float(w_in-1) * (xyz[(point_pos) * 3 + 0] + 1.0)) / 2.0);
            break;
        case 1:
            sample_grid_thisjob[threadIdx.x] = ((float(h_in-1) * (xyz[(point_pos) * 3 + 1] + 1.0)) / 2.0);
            break;
        case 2:
            sample_grid_thisjob[threadIdx.x] = ((float(line_h_in-1) * (xyz[(point_pos) * 3 + 2] + 1.0)) / 2.0);
            break;
      }
    }
  }

  __syncthreads();
  for (int nanoit = 0; nanoit < nanobatch; nanoit++) {
    int point_pos = point_pos_base + nanoit;
    if (point_pos >= xyz_num) {
      break;
    }
    int ib = b[point_pos];
    long b_pos = long(ib) * h_in * w_in * C;

    float ix = sample_grid_thisjob[nanoit * 3 + 0];
    float iy = sample_grid_thisjob[nanoit * 3 + 1];
    float iz = sample_grid_thisjob[nanoit * 3 + 2];

    int32_t iz_nw = int32_t(iz);
    int ix_nw = (int)floorf(ix);
    int iy_nw = (int)floorf(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);

    int channel_offset = (int)threadIdx.x;
    // long channel_pos = long(channel_offset) * block_in * h_in * w_in;

    float nw_val  = plane[b_pos + (iy_nw * w_in + ix_nw) * C + channel_offset];
    float ne_val  = plane[b_pos + (iy_nw * w_in + min((ix_nw + 1), (w_in-1))) * C + channel_offset] ;
    float sw_val  = plane[b_pos + (min((iy_nw + 1), (h_in-1)) * w_in + ix_nw) * C + channel_offset] ;
    float se_val  = plane[b_pos + (min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))) * C + channel_offset] ;

    float plane_coef_tmp = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;


    float line_coef_tmp = ((line[channel_offset * line_h_in * line_w_in + iz_nw * line_w_in + ib] * (float(iz_nw + 1) - iz))) +
          ((line[channel_offset * line_h_in * line_w_in + min((iz_nw + 1), (line_h_in-1)) * line_w_in + ib] * (iz - float(iz_nw))));

    sigma[point_pos * C * plane_in + plane_idx * C + channel_offset] =  plane_coef_tmp * line_coef_tmp;
  }
}
// 与xyzb版本不同点在于，xyz矩阵与b矩阵分离，且xyb
__global__ void fusedkernel_serial_xyb_bz_nbhwc2_outhalf(const float* xyz, const int* b, float* plane, const float* line, half* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int C, const int nanobatch, const int plane_idx, const int plane_in) {
  __shared__ float sample_grid_shared[8 * 8 * 10];  // 一个block上有8个job组，每组有8个有效点数，每个有效点数需要10个float空间存数

  float *sample_grid_thisjob = sample_grid_shared + threadIdx.y * 80;

  int job_id = blockIdx.x * blockDim.y + threadIdx.y;  // blockDim.y == job_num_per_block
  int point_pos_base = job_id * nanobatch;
  // 一个job组中的一些变量
  if(threadIdx.x < nanobatch * 3){
    int nanoit   = threadIdx.x / 3;
    int read_idx = threadIdx.x % 3;
    int point_pos = point_pos_base + nanoit;
    if (point_pos < xyz_num) {
      switch (read_idx) {
        case 0:
            sample_grid_thisjob[threadIdx.x] = ((float(w_in-1) * (xyz[(point_pos) * 3 + 0] + 1.0)) / 2.0);
            break;
        case 1:
            sample_grid_thisjob[threadIdx.x] = ((float(h_in-1) * (xyz[(point_pos) * 3 + 1] + 1.0)) / 2.0);
            break;
        case 2:
            sample_grid_thisjob[threadIdx.x] = ((float(line_h_in-1) * (xyz[(point_pos) * 3 + 2] + 1.0)) / 2.0);
            break;
      }
    }
  }

  __syncthreads();
  for (int nanoit = 0; nanoit < nanobatch; nanoit++) {
    int point_pos = point_pos_base + nanoit;
    if (point_pos >= xyz_num) {
      break;
    }
    int ib = b[point_pos];
    long b_pos = long(ib) * h_in * w_in * C;

    float ix = sample_grid_thisjob[nanoit * 3 + 0];
    float iy = sample_grid_thisjob[nanoit * 3 + 1];
    float iz = sample_grid_thisjob[nanoit * 3 + 2];

    int32_t iz_nw = int32_t(iz);
    int ix_nw = (int)floorf(ix);
    int iy_nw = (int)floorf(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);

    int channel_offset = (int)threadIdx.x;
    // long channel_pos = long(channel_offset) * block_in * h_in * w_in;

    float nw_val  = plane[b_pos + (iy_nw * w_in + ix_nw) * C + channel_offset];
    float ne_val  = plane[b_pos + (iy_nw * w_in + min((ix_nw + 1), (w_in-1))) * C + channel_offset] ;
    float sw_val  = plane[b_pos + (min((iy_nw + 1), (h_in-1)) * w_in + ix_nw) * C + channel_offset] ;
    float se_val  = plane[b_pos + (min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))) * C + channel_offset] ;

    float plane_coef_tmp = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;


    float line_coef_tmp = ((line[channel_offset * line_h_in * line_w_in + iz_nw * line_w_in + ib] * (float(iz_nw + 1) - iz))) +
          ((line[channel_offset * line_h_in * line_w_in + min((iz_nw + 1), (line_h_in-1)) * line_w_in + ib] * (iz - float(iz_nw))));

    sigma[point_pos * C * plane_in + plane_idx * C + channel_offset] =  __float2half(plane_coef_tmp * line_coef_tmp);
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
// fusedkernel_serial_xyb_bz_144共修改了三版：此处仅支持第三版（性能最优
// 1. 仅修改输出按144连续的版本, plane_idx标识该次计算的是第几个, plane_in标识plane的数目, 为3
// 2. 在上版本基础上, 调了block与grid的设置, 在blockid.x维度又加回串行化
// 3. 在上版本基础上, 循环下放
__global__ void fusedkernel_serial_xyb_bz_144_3(const float* xyz, const int* b, const float* plane, const float* line, float* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int channel_in, const int serial, const int plane_idx, const int plane_in, int tid_y_max, int num_block) {
  if (((int)threadIdx.y < (min(((-num_block * (int)blockIdx.y) + (xyz_num-1)), (num_block-1)) + 1)) && threadIdx.x < tid_y_max) {
    int ib = b[((int)blockIdx.y * num_block) + (int)threadIdx.y];
    float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * num_block) + (int)threadIdx.y) * 3 + 0] + 1.0)) / 2.0);
    float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * num_block) + (int)threadIdx.y) * 3 + 1] + 1.0)) / 2.0);
    float iz = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * num_block) + (int)threadIdx.y) * 3 + 2] + 1.0)) / 2.0);
    int ix_nw = (int)floorf(ix);
    int iy_nw = (int)floorf(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);
    int32_t iz_nw = int32_t(iz);

    for (int cc = 0; cc < serial; cc++) {
      int channel_offset = (int)blockIdx.x * tid_y_max * serial + cc * tid_y_max + threadIdx.x;

      long channel_offset_l = long(channel_offset);
      float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ?
        plane[IDX4C(ix_nw, iy_nw, ib ,channel_offset_l, w_in, h_in, block_in) ] : 0;
      float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ?
        plane[IDX4C(min((ix_nw + 1), (w_in-1)), iy_nw, ib ,channel_offset_l, w_in, h_in, block_in) ] : 0;
      float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ?
        plane[IDX4C(ix_nw, min((iy_nw + 1), (h_in-1)), ib ,channel_offset_l, w_in, h_in, block_in) ] : 0;
      float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ?
        plane[IDX4C(min((ix_nw + 1), (w_in-1)), min((iy_nw + 1), (h_in-1)), ib ,channel_offset_l, w_in, h_in, block_in) ] : 0;
      float plane_coef_tmp = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;

      float line_coef_tmp = ((line[IDX3C(ib, iz_nw, channel_offset, line_w_in, line_h_in)] * (float(iz_nw + 1) - iz))) +
          ((line[IDX3C(ib, min((iz_nw + 1), (line_h_in-1)), channel_offset, line_w_in, line_h_in)] * (iz - float(iz_nw))));

      sigma[(((int)blockIdx.y * num_block) + (int)threadIdx.y) * plane_in * channel_in + plane_idx * channel_in + channel_offset] = plane_coef_tmp * line_coef_tmp;
    }
  }
}

// 同上，是outputhalf版本
__global__ void fusedkernel_serial_xyb_bz_144_3_outhalf(const float* xyz, const int* b, const float* plane, const float* line, half* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int channel_in, const int serial, const int plane_idx, const int plane_in, int tid_y_max, int num_block) {
  if (((int)threadIdx.y < (min(((-num_block * (int)blockIdx.y) + (xyz_num-1)), (num_block-1)) + 1)) && threadIdx.x < tid_y_max) {
    int ib = b[((int)blockIdx.y * num_block) + (int)threadIdx.y];
    float ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * num_block) + (int)threadIdx.y) * 3 + 0] + 1.0)) / 2.0);
    float iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * num_block) + (int)threadIdx.y) * 3 + 1] + 1.0)) / 2.0);
    float iz = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * num_block) + (int)threadIdx.y) * 3 + 2] + 1.0)) / 2.0);
    int ix_nw = (int)floorf(ix);
    int iy_nw = (int)floorf(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);
    int32_t iz_nw = int32_t(iz);

    for (int cc = 0; cc < serial; cc++) {
      int channel_offset = (int)blockIdx.x * tid_y_max * serial + cc * tid_y_max + threadIdx.x;

      long channel_offset_l = long(channel_offset);
      float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ?
        plane[IDX4C(ix_nw, iy_nw, ib ,channel_offset_l, w_in, h_in, block_in) ] : 0;
      float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ?
        plane[IDX4C(min((ix_nw + 1), (w_in-1)), iy_nw, ib ,channel_offset_l, w_in, h_in, block_in) ] : 0;
      float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ?
        plane[IDX4C(ix_nw, min((iy_nw + 1), (h_in-1)), ib ,channel_offset_l, w_in, h_in, block_in) ] : 0;
      float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ?
        plane[IDX4C(min((ix_nw + 1), (w_in-1)), min((iy_nw + 1), (h_in-1)), ib ,channel_offset_l, w_in, h_in, block_in) ] : 0;
      float plane_coef_tmp = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;

      float line_coef_tmp = ((line[IDX3C(ib, iz_nw, channel_offset, line_w_in, line_h_in)] * (float(iz_nw + 1) - iz))) +
          ((line[IDX3C(ib, min((iz_nw + 1), (line_h_in-1)), channel_offset, line_w_in, line_h_in)] * (iz - float(iz_nw))));

      sigma[(((int)blockIdx.y * num_block) + (int)threadIdx.y) * plane_in * channel_in + plane_idx * channel_in + channel_offset] = __float2half(plane_coef_tmp * line_coef_tmp);
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





int compute_grid_sample_and_ewproduct_serial_xyb_bz_144_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial, int plane_idx, int plane_in, int trymode, int tryvalue, int num_block)
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


    if (trymode == 3){
      const dim3 block(tryvalue, num_block, 1);
      const dim3 grid(c / tryvalue / serial, int((xyz_num - 1) / num_block) + 1, 1);
      AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_xyb_bz_cuda", ([&] {
      fusedkernel_serial_xyb_bz_144_3<<<grid, block>>>(
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
          serial,
          plane_idx,
          plane_in,
          tryvalue,
          num_block);
      }));
    }

    return 0;
}


int compute_grid_sample_and_ewproduct_serial_xyb_bz_144_outhalf_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial, int plane_idx, int plane_in, int trymode, int tryvalue, int num_block)
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


    if (trymode == 3){
      const dim3 block(tryvalue, num_block, 1);
      const dim3 grid(c / tryvalue / serial, int((xyz_num - 1) / num_block) + 1, 1);
      AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_xyb_bz_cuda", ([&] {
      fusedkernel_serial_xyb_bz_144_3_outhalf<<<grid, block>>>(
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
          serial,
          plane_idx,
          plane_in,
          tryvalue,
          num_block);
      }));
    }

    return 0;
}
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

// 尝试加速, 使用循环下放, 删去边界判断
__global__ void fusedkernel_sum_xyz_b_2(const float* xyz, const int* b, const float* plane, const float* line, float* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int C, bool judgeOverflow, int num_block) {
  if (((int)threadIdx.x < (min(((-num_block * (int)blockIdx.x) + (xyz_num-1)), (num_block-1)) + 1))) {
    float sum;
    sum = 0.0;

    unsigned int xyzlen = 3; //4
    int ib = b[((int)blockIdx.x * num_block) + (int)threadIdx.x];
    float ix, iy, iz;
    float result_plane, result_line;

    ix = ((float(w_in-1) * (xyz[(((int)blockIdx.x * num_block) + (int)threadIdx.x) * xyzlen + 0] + 1.0)) / 2.0);
    iy = ((float(h_in-1) * (xyz[(((int)blockIdx.x * num_block) + (int)threadIdx.x) * xyzlen + 1] + 1.0)) / 2.0);
    long b_pos = long(ib) * h_in * w_in;
    int ix_nw = (int)floorf(ix);
    int iy_nw = (int)floorf(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);

    iz = ((float(line_h_in-1) * (xyz[(((int)blockIdx.x * num_block) + (int)threadIdx.x) * xyzlen + 2] + 1.0)) / 2.0);
    int32_t iz_nw = int32_t(iz);

    for (int cc = 0; cc < C; cc++) {
      if (judgeOverflow && (abs(xyz[(((int)blockIdx.x * num_block) + (int)threadIdx.x) * xyzlen + 0]) > 1.0 || abs(xyz[(((int)blockIdx.x * num_block) + (int)threadIdx.x) * xyzlen + 1]) > 1.0)){
        result_plane = 0.0;
      }else{


        long channel_pos = long(cc) * block_in * h_in * w_in;
        float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ?
          plane[channel_pos + b_pos + iy_nw * w_in + ix_nw ] : 0;
        float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ?
          plane[channel_pos + b_pos + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] : 0;
        float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ?
          plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] : 0;
        float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ?
          plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] : 0;
        result_plane = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
      }

      if (judgeOverflow && (abs(xyz[(((int)blockIdx.x * num_block) + (int)threadIdx.x) * xyzlen + 2]) > 1.0 || abs(xyz[(((int)blockIdx.x * num_block) + (int)threadIdx.x) * xyzlen + 3]) > 1.0)){
        result_line = 0.0;
      }else{


        result_line = ((line[cc * line_h_in * line_w_in + iz_nw * line_w_in + ib] * (float(iz_nw + 1) - iz))) +
           ((line[cc * line_h_in * line_w_in + min((iz_nw + 1), (line_h_in-1)) * line_w_in + ib] * (iz - float(iz_nw))));
      }
      sum += result_plane * result_line;
    }
    sigma[(((int)blockIdx.x * num_block) + (int)threadIdx.x)] = sum;
  }
}

__global__ void tryp
(const int* b) {

}


// 在kernel层面上融合, 一个thread算3个plane并直接求和
__global__ void fusedkernel_sum_xyz_b_3
(const float* xyz, const int* b, float** plane_line_ptr_not_share, float* sigma,
  const int xyz_num, const int block_in, const int* hw_in_not_share, const int C, const int arraySize, bool judgeOverflow, int num_block) {
    // 给参数数组开一个sharedmemory空间, 其中10为arraySize能取到的最大值.
    __shared__ int hw_in[10 * 4];
    __shared__ float* plane_line_ptr[10 * 2];
    int tid = (int)threadIdx.x;
    if (tid < arraySize * 4){
        hw_in[tid] = hw_in_not_share[tid];
    }else if(tid < arraySize * 4 + arraySize * 2){
        plane_line_ptr[tid - arraySize * 4] = plane_line_ptr_not_share[tid - arraySize * 4];
    }
    __syncthreads();

  if (((int)threadIdx.x < (min(((- num_block  * (int)blockIdx.x) + (xyz_num-1)), (num_block - 1)) + 1))) {
    float sum;
    sum = 0.0;
    unsigned int xyzlen = 3; //4
    int ib = b[((int)blockIdx.x * num_block) + (int)threadIdx.x];
    float ix, iy, iz;
    float result_plane, result_line;
    int m_id = (((int)blockIdx.x * num_block) + (int)threadIdx.x);
    // 遍历每个分辨率
    for (int planeid = 0; planeid < arraySize; planeid++){
      // hw_in按照h_in, w_in, line_h_in, line_w_in...存放
      // 按典型数值:plane:1x16x24x493x920, line: 1x16x310x24, 对应下放四个数值为:
                      // IDX2C(param_id, plane_id, param_len), 其中param_len =  4
      const int h_in = hw_in[planeid * 4 + 0];      // 493
      const int w_in = hw_in[planeid * 4 + 1];      // 920
      const int line_h_in = hw_in[planeid * 4 + 2]; // 310
      const int line_w_in = hw_in[planeid * 4 + 3]; // 24
      // plane_line_ptr按照plane_ptr, line_ptr...存放
                                          // IDX2C(ptr_id, plane_id, ptr_len), 其中ptr_len = 2
      const float * plane = plane_line_ptr[planeid * 2 + 0];
      const float * line = plane_line_ptr[planeid * 2 + 1];

                             // IDX2C(xyz_id, m_pos, xyzlen), 其中m_pos = (((int)blockIdx.x * num_block) + (int)threadIdx.x)
      ix = ((float(w_in-1) * (xyz[m_id * xyzlen + 0] + 1.0)) / 2.0);
      iy = ((float(h_in-1) * (xyz[m_id * xyzlen + 1] + 1.0)) / 2.0);
      long b_pos = long(ib) * h_in * w_in;
      int ix_nw = (int)floorf(ix);
      int iy_nw = (int)floorf(iy);
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;
      float nw = (ix_se - ix) * (iy_se - iy);
      float ne = (ix - ix_sw) * (iy_sw - iy);
      float sw = (ix_ne - ix) * (iy - iy_ne);
      float se = (ix - ix_nw) * (iy - iy_nw);

      iz = ((float(line_h_in-1) * (xyz[m_id * xyzlen + 2] + 1.0)) / 2.0);
      int32_t iz_nw = int32_t(iz);

      // 遍历每个channel
      for (int cc = 0; cc < C; cc++) {
        if (judgeOverflow && (abs(xyz[m_id * xyzlen + 0]) > 1.0 || abs(xyz[m_id * xyzlen + 1]) > 1.0)){
          result_plane = 0.0;
        }else{
          long channel_pos = long(cc) * block_in * h_in * w_in;
          float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ?
            // IDX4C(ix_nw, iy_nw, ib ,cc, w_in, h_in, block_in)
            plane[channel_pos + b_pos + iy_nw * w_in + ix_nw ] : 0;
          float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ?
            // IDX4C(min((ix_nw + 1), (w_in-1)), iy_nw, ib ,cc, w_in, h_in, block_in)
            plane[channel_pos + b_pos + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] : 0;
          float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ?
            // IDX4C(ix_nw, min((iy_nw + 1), (h_in-1)), ib ,cc, w_in, h_in, block_in)
            plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] : 0;
          float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ?
            // IDX4C(min((ix_nw + 1), (w_in-1)), min((iy_nw + 1), (h_in-1)), ib ,cc, w_in, h_in, block_in)
            plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] : 0;
          result_plane = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
        }

        if (judgeOverflow && (abs(xyz[m_id * xyzlen + 2]) > 1.0 || abs(xyz[m_id * xyzlen + 3]) > 1.0)){
          result_line = 0.0;
        }else{                // IDX3C(ib, iz_nw, cc, line_w_in, line_h_in)
          result_line = ((line[cc * line_h_in * line_w_in + iz_nw * line_w_in + ib] * (float(iz_nw + 1) - iz))) +
            ((line[cc * line_h_in * line_w_in + min((iz_nw + 1), (line_h_in-1)) * line_w_in + ib] * (iz - float(iz_nw))));
        }
        sum += result_plane * result_line;
      }
    }
    sigma[m_id] = sum;
  }
}


// 在kernel层面上融合, 一个thread算3个plane并直接求和, 计算使用nbhwc作为输入，输出为[c]
__global__ void fusedkernel_sum_xyz_b_3_nbhwc
(const float* xyz, const int* b, float** plane_line_ptr_not_share, float* sigma,
  const int xyz_num, const int block_in, const int* hw_in_not_share, const int C, const int arraySize, bool judgeOverflow, int num_block) {
    // 给参数数组开一个sharedmemory空间, 其中10为arraySize能取到的最大值.
    __shared__ int hw_in[10 * 4];
    __shared__ float* plane_line_ptr[10 * 2];
    int tid = (int)threadIdx.x;
    if (tid < arraySize * 4){
        hw_in[tid] = hw_in_not_share[tid];
    }else if(tid < arraySize * 4 + arraySize * 2){
        plane_line_ptr[tid - arraySize * 4] = plane_line_ptr_not_share[tid - arraySize * 4];
    }
    __syncthreads();

    int float4_len = 4;
    int point_num = 4;
    int shared_per_thread = float4_len * point_num;
    // 一个sm上的num_block个thread，每个thread使用4个float4的空间
    __shared__ float plane_channel[10000];

  if (((int)threadIdx.x < (min(((- num_block  * (int)blockIdx.x) + (xyz_num-1)), (num_block - 1)) + 1))) {
    float sum;
    sum = 0.0;
    unsigned int xyzlen = 3; //4
    int ib = b[((int)blockIdx.x * num_block) + (int)threadIdx.x];
    float ix, iy, iz;
    float result_plane, result_line;
    int m_id = (((int)blockIdx.x * num_block) + (int)threadIdx.x);
    // 遍历每个分辨率
    for (int planeid = 0; planeid < arraySize; planeid++){
      // hw_in按照h_in, w_in, line_h_in, line_w_in...存放
      // 按典型数值:plane:1x16x24x493x920, line: 1x16x310x24, 对应下放四个数值为:
                      // IDX2C(param_id, plane_id, param_len), 其中param_len =  4
      const int h_in = hw_in[planeid * 4 + 0];      // 493
      const int w_in = hw_in[planeid * 4 + 1];      // 920
      const int line_h_in = hw_in[planeid * 4 + 2]; // 310
      const int line_w_in = hw_in[planeid * 4 + 3]; // 24
      // plane_line_ptr按照plane_ptr, line_ptr...存放
                                          // IDX2C(ptr_id, plane_id, ptr_len), 其中ptr_len = 2
      float * plane = plane_line_ptr[planeid * 2 + 0];
      float4* plane4 = reinterpret_cast<float4*>(plane);
      const float * line = plane_line_ptr[planeid * 2 + 1];

                             // IDX2C(xyz_id, m_pos, xyzlen), 其中m_pos = (((int)blockIdx.x * num_block) + (int)threadIdx.x)
      ix = ((float(w_in-1) * (xyz[m_id * xyzlen + 0] + 1.0)) / 2.0);
      iy = ((float(h_in-1) * (xyz[m_id * xyzlen + 1] + 1.0)) / 2.0);
      int ix_nw = (int)floorf(ix);
      int iy_nw = (int)floorf(iy);
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;
      float nw = (ix_se - ix) * (iy_se - iy);
      float ne = (ix - ix_sw) * (iy_sw - iy);
      float sw = (ix_ne - ix) * (iy - iy_ne);
      float se = (ix - ix_nw) * (iy - iy_nw);

      iz = ((float(line_h_in-1) * (xyz[m_id * xyzlen + 2] + 1.0)) / 2.0);
      int32_t iz_nw = int32_t(iz);

      if (judgeOverflow && (abs(xyz[m_id * xyzlen + 0]) > 1.0 || abs(xyz[m_id * xyzlen + 1]) > 1.0)){
        continue;
      }



      int C_outer = C / float4_len;
      int base_index = 0;
      for (int cc_outer = 0; cc_outer < C_outer; cc_outer++) {
        base_index = (int)threadIdx.x * point_num;
        ((float4*)plane_channel)[base_index + 0] =
        plane4[(long(ib) * h_in * w_in + iy_nw * w_in + ix_nw) * C_outer + cc_outer];
        ((float4*)plane_channel)[base_index + 1] =
        plane4[(long(ib) * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))) * C_outer + cc_outer];
        ((float4*)plane_channel)[base_index + 2] =
        plane4[(long(ib) * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw) * C_outer + cc_outer];
        ((float4*)plane_channel)[base_index + 3] =
        plane4[(long(ib) * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))) * C_outer + cc_outer];


        for(int cc_inner = 0; cc_inner < float4_len; cc_inner++){
          if (judgeOverflow && (abs(xyz[m_id * xyzlen + 0]) > 1.0 || abs(xyz[m_id * xyzlen + 1]) > 1.0)){
            result_plane = 0.0;
          }else{
            base_index = (int)threadIdx.x * shared_per_thread + cc_inner;
            float nw_val  = plane_channel[ base_index + 0 * float4_len ];
            float ne_val  = plane_channel[ base_index + 1 * float4_len ];
            float sw_val  = plane_channel[ base_index + 2 * float4_len ];
            float se_val  = plane_channel[ base_index + 3 * float4_len ];
            result_plane = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
          }

          if (judgeOverflow && (abs(xyz[m_id * xyzlen + 2]) > 1.0 || abs(xyz[m_id * xyzlen + 3]) > 1.0)){
            result_line = 0.0;
          }else{                // IDX3C(ib, iz_nw, cc, line_w_in, line_h_in)
            int cc = cc_outer * float4_len + cc_inner;
            result_line = ((line[cc * line_h_in * line_w_in + iz_nw * line_w_in + ib] * (float(iz_nw + 1) - iz))) +
              ((line[cc * line_h_in * line_w_in + min((iz_nw + 1), (line_h_in-1)) * line_w_in + ib] * (iz - float(iz_nw))));
          }
          sum += result_plane * result_line;

        }

      }


    }
    sigma[m_id] = sum;
  }
}






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
    float ix = grid_sampler_compute_source_index(x, w_in, false, true);
    float iy = grid_sampler_compute_source_index(y, h_in, false, true);
    ix = safe_downgrade_to_int_range(ix);
    iy = safe_downgrade_to_int_range(iy);
    plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = gridsample2d_cal(plane, h_in, w_in, ix, iy, (int)blockIdx.x);

  }
}
template <typename scalar_t>
__global__ void kernel2d_nhwc_in_out(const scalar_t *__restrict__ xyz,
const scalar_t *__restrict__ plane,  scalar_t *__restrict__ plane_coef, /*const float* line, float* line_coef, */
const int xyz_num, const int channel_num, const int h_in, const int w_in  /*, const int line_h_in, const int line_w_in*/) {
  int pt_block_idx = NUM_BLOCK_64 * (int)blockIdx.x;
  int channel_offset = (int)threadIdx.x;
  if (((int)threadIdx.y < (min(((-pt_block_idx) + (xyz_num-1)), (NUM_BLOCK_64-1)) + 1))) {
    scalar_t x = xyz[(pt_block_idx + (int)threadIdx.y) * 3 + 0];
    scalar_t y = xyz[(pt_block_idx + (int)threadIdx.y) * 3 + 1];
    float ix = grid_sampler_compute_source_index(x, w_in, false, true);
    float iy = grid_sampler_compute_source_index(y, h_in, false, true);
    ix = safe_downgrade_to_int_range(ix);
    iy = safe_downgrade_to_int_range(iy);
    plane_coef[(pt_block_idx + (int)threadIdx.y) * channel_num + channel_offset] = gridsample2d_nhwc_input_cal(plane, h_in, w_in, channel_num, ix, iy, channel_offset);
  }
}

template <typename scalar_t>
__global__ void kernel2d_nhwc_shared(const scalar_t *__restrict__ xyz,
scalar_t *__restrict__ plane,  scalar_t *__restrict__ plane_coef, /*const float* line, float* line_coef, */
const int xyz_num, const int channel_num, const int h_in, const int w_in  /*, const int line_h_in, const int line_w_in*/) {
  // continuous on channel
  __shared__ float plane_channel[NUM_BLOCK_64 * 4 * 16];
  // float4
  float4* plane4 = reinterpret_cast<float4*>(plane);
  int pt_block_idx = NUM_BLOCK_64 * (int)blockIdx.x;
  int channel_offset = (int)threadIdx.x;
  int pt_group = (int)threadIdx.x / 4;
  int channel_group = (int)threadIdx.x % 4;

  if (((int)threadIdx.y < (min(((-pt_block_idx) + (xyz_num-1)), (NUM_BLOCK_64-1)) + 1))) {
    scalar_t x = xyz[(pt_block_idx + (int)threadIdx.y) * 3 + 0];
    scalar_t y = xyz[(pt_block_idx + (int)threadIdx.y) * 3 + 1];
    float ix = grid_sampler_compute_source_index(x, w_in, false, true);
    float iy = grid_sampler_compute_source_index(y, h_in, false, true);
    ix = safe_downgrade_to_int_range(ix);
    iy = safe_downgrade_to_int_range(iy);

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


// plane_coef[(pt_block_idx + (int)threadIdx.y) * channel_num + channel_offset] = gridsample2d_nhwc_input_cal(plane, h_in, w_in, channel_num, ix, iy, channel_offset);
    float4 zero4 = {0, 1, 2, 3};
    //  plane[(iy_nw * w_in + ix_nw) * channel_num + channel_offset]
    // ((float4*)plane_channel)[(int)threadIdx.y * 4 * channel_num + pt_group * channel_num + 0 + (channel_group * 4)] = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ? ((float4*)plane4)[((iy_nw + 1 - (pt_group / 2)) * w_in + ix_nw + 1 - (pt_group % 2)) * channel_num + 0 + (channel_group * 4)] : zero4;
    ((float4*)plane_channel)[(int)threadIdx.y * channel_num + pt_group * channel_num / 4 + (channel_group)] =
    plane4[((iy_nw + 1 - (pt_group / 2)) * w_in + ix_nw + 1 - (pt_group % 2)) * channel_num / 4 + (channel_group)];

    __syncthreads();
    float nw_val = ((float*)plane_channel)[(int)threadIdx.y * 4 * channel_num + 3 * channel_num + channel_offset];
    float ne_val = ((float*)plane_channel)[(int)threadIdx.y * 4 * channel_num + 2 * channel_num + channel_offset];
    float sw_val = ((float*)plane_channel)[(int)threadIdx.y * 4 * channel_num + 1 * channel_num + channel_offset];
    float se_val = ((float*)plane_channel)[(int)threadIdx.y * 4 * channel_num + 0 * channel_num + channel_offset];

    // __syncthreads();

    plane_coef[(pt_block_idx + (int)threadIdx.y) * channel_num + channel_offset] = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
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


int compute_grid_sample_and_ewproduct_serial_xyb_bz_nbhwc_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);
    // plane shape (N,C,h_in,w_in)  [1, 9, 688, 688, 16]
    const int n = plane.size(0); // n == 1
    const int block_in = plane.size(1); // block_in == 8
    const int h_in = plane.size(2); // h_in == 2047
    const int w_in = plane.size(3); // w_in == 2047
    const int c = plane.size(4); // c == 16

    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    assert( c % serial == 0);
    assert(block_in==line_w_in);

    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid(c / serial, int((xyz_num - 1) / NUM_BLOCK) + 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_xyb_bz_nbhwc_cuda", ([&] {
    fusedkernel_serial_xyb_bz_nbhwc<<<grid, block>>>(
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

int compute_grid_sample_and_ewproduct_serial_xyb_bz_nbhwc2_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial, int plane_idx, int plane_in)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);
    // plane shape (N,C,h_in,w_in)  [1, 9, 688, 688, 16]
    const int n = plane.size(0); // n == 1
    const int block_in = plane.size(1); // block_in == 8
    const int h_in = plane.size(2); // h_in == 2047
    const int w_in = plane.size(3); // w_in == 2047
    const int c = plane.size(4); // c == 16

    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    // assert( c % serial == 0);
    assert(block_in==line_w_in);


    int nanobatch = serial;                 // 一簇threadx完成nanobatch个有效点数的计算，其中每个threadidx对应完成该channel的计算。我们认为这一簇算是一个job，
    int job_num_per_block = 8;              // 一共有 m / nano个 job

    const dim3 grid(int((xyz_num - 1) / (nanobatch * job_num_per_block)) + 1, 1, 1);
    const dim3 block(c, job_num_per_block, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_xyb_bz_nbhwc2_cuda", ([&] {
    fusedkernel_serial_xyb_bz_nbhwc2<<<grid, block>>>(
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
        nanobatch, plane_idx, plane_in);
    }));
    return 0;
}

int compute_grid_sample_and_ewproduct_serial_xyb_bz_nbhwc2_outhalf_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial, int plane_idx, int plane_in)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);
    // plane shape (N,C,h_in,w_in)  [1, 9, 688, 688, 16]
    const int n = plane.size(0); // n == 1
    const int block_in = plane.size(1); // block_in == 8
    const int h_in = plane.size(2); // h_in == 2047
    const int w_in = plane.size(3); // w_in == 2047
    const int c = plane.size(4); // c == 16

    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    // assert( c % serial == 0);
    assert(block_in==line_w_in);


    int nanobatch = serial;                 // 一簇threadx完成nanobatch个有效点数的计算，其中每个threadidx对应完成该channel的计算。我们认为这一簇算是一个job，
    int job_num_per_block = 8;              // 一共有 m / nano个 job

    const dim3 grid(int((xyz_num - 1) / (nanobatch * job_num_per_block)) + 1, 1, 1);
    const dim3 block(c, job_num_per_block, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_ewproduct_serial_xyb_bz_nbhwc2_cuda", ([&] {
    fusedkernel_serial_xyb_bz_nbhwc2_outhalf<<<grid, block>>>(
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
        nanobatch, plane_idx, plane_in);
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
  torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int num_block)
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

    const dim3 block(num_block, 1, 1);
    const dim3 grid(int((xyz_num - 1) / num_block) + 1, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_sum_xyz_b_cuda", ([&] {
    fusedkernel_sum_xyz_b_2<<<grid, block>>>(
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
        false, num_block);
    }));
    return 0;
}

// 不再维护
// int compute_grid_sample_and_sum_xyz_b3_cuda(
//   torch::Tensor xyz, torch::Tensor b, std::vector<torch::Tensor> plane,
//   std::vector<torch::Tensor> line, torch::Tensor sigma, int num_block)
// {
//     const int xyz_num = xyz.size(0);

//     // plane shape (N,C,h_in,w_in)
//     const int n = plane[0].size(0); // n == 1
//     const int c = plane[0].size(1); // c == 16
//     const int block_in = plane[0].size(2); // block_in == 8

//     // 查看cudamemorycpy的耗时
//     // cudaEvent_t start, stop1;
//     // cudaEventCreate(&start);
//     // cudaEventCreate(&stop1);
//     // cudaEventRecord(start);

//     // 初始化参数数组, 在cpu端, 这些参数数组
//     int arraySize = plane.size();    // tensorf.density_plane的长度,只测试过为3的情况, 待补充为2情况的测试
//     int h_in[arraySize];             // 各个plane平面的高度
//     int w_in[arraySize];             // 各个plane平面的宽度
//     int line_h_in[arraySize];        // 各个line平面的高度
//     int line_w_in[arraySize];        // 各个line平面的宽度
//     float* plane_l[arraySize];       // 各个plane平面数据的指针
//     float* line_l[arraySize];        // 各个line平面数据的指针

//     // 赋值参数数组, 在cpu端
//     for(int idx = 0; idx < plane.size(); idx++ ){// 以plane: [torch.float32 of size 1x16x8x343x172]
//                                                  //    line: [torch.float32 of size 1x16x57x8] 为例子
//       h_in[idx] = plane[idx].size(2 + 1);        // 343 of 1x16x8x343x172
//       w_in[idx] = plane[idx].size(3 + 1);        // 172 of 1x16x8x343x172
//       line_h_in[idx] = line[idx].size(2);        // 57  of 1x16x57x8
//       line_w_in[idx] = line[idx].size(3);        // 8   of 1x16x57x8
//       plane_l[idx] = plane[idx].data<float>();   // ptr -> 1x16x8x343x172
//       line_l[idx] = line[idx].data<float>();     // ptr -> 1x16x57x8
//     }

//     // 拷贝参数数组到gpu端, 先依次申请GPU内存空间
//     int* h_in_c, *w_in_c, *line_h_in_c, *line_w_in_c;
//     float** plane_c;
//     float** line_c;
//     cudaMalloc((void**)&h_in_c, arraySize * sizeof(int));
//     cudaMalloc((void**)&w_in_c, arraySize * sizeof(int));
//     cudaMalloc((void**)&line_h_in_c, arraySize * sizeof(int));
//     cudaMalloc((void**)&line_w_in_c, arraySize * sizeof(int));
//     cudaMalloc((void**)&plane_c, arraySize * sizeof(float*));
//     cudaMalloc((void**)&line_c, arraySize * sizeof(float*));
//     // 再依次拷贝
//     cudaMemcpy(h_in_c, h_in, arraySize * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(w_in_c, w_in, arraySize * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(line_h_in_c, line_h_in, arraySize * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(line_w_in_c, line_w_in, arraySize * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(plane_c, plane_l, arraySize * sizeof(float*), cudaMemcpyHostToDevice);
//     cudaMemcpy(line_c, line_l, arraySize * sizeof(float*), cudaMemcpyHostToDevice);


//     // 查看cudamemorycpy的耗时
//     // cudaEventRecord(stop1);
//     // cudaDeviceSynchronize();
//     // float Time1;
//     // cudaEventElapsedTime(&Time1, start, stop1);
//     // std::cout << "    cudamemorycpy use time \t"  << Time1<< " ms\n";

//     //原本非自适应的参数传递
//     // const int h_in0 = plane0.size(2 + 1); // h_in == 2047
//     // const int w_in0 = plane0.size(3 + 1); // w_in == 2047
//     // const int line_h_in0 = line0.size(2); // line_h_in == 255
//     // const int line_w_in0 = line0.size(3); // line_w_in == 1
//     // const int h_in1 = plane1.size(2 + 1); // h_in == 2047
//     // const int w_in1 = plane1.size(3 + 1); // w_in == 2047
//     // const int line_h_in1 = line1.size(2); // line_h_in == 255
//     // const int line_w_in1 = line1.size(3); // line_w_in == 1
//     // const int h_in2 = plane2.size(2 + 1); // h_in == 2047
//     // const int w_in2 = plane2.size(3 + 1); // w_in == 2047
//     // const int line_h_in2 = line2.size(2); // line_h_in == 255
//     // const int line_w_in2 = line2.size(3); // line_w_in == 1


//     const dim3 block(num_block, 1, 1);
//     const dim3 grid(int((xyz_num - 1) / num_block) + 1, 1, 1);
        // AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_sum_xyz_b_3_cuda", ([&] {
        // fusedkernel_sum_xyz_b_3<<<grid, block>>>(
        //     xyz.data<float>(),
        //     b.data<int>(),
        //     plane_c,
        //     line_c,
        //     sigma.data<float>(),
        //     xyz_num,
        //     block_in,
        //     h_in_c,
        //     w_in_c,
        //     line_h_in_c,
        //     line_w_in_c,
        //     c, arraySize,
        //     false, num_block);
        // }));

//     cudaFree(plane_c);
//     cudaFree(line_c);
//     cudaFree(h_in_c);
//     cudaFree(w_in_c);
//     cudaFree(line_h_in_c);
//     cudaFree(line_w_in_c);
//     return 0;
// }


int compute_grid_sample_and_sum_xyz_b3_noMalloc_cuda(
  torch::Tensor xyz, torch::Tensor b, std::vector<torch::Tensor> plane, std::vector<torch::Tensor> line,
  torch::Tensor hw_in, torch::Tensor plane_line_ptr,
  torch::Tensor sigma, int num_block)
{
    const int xyz_num = xyz.size(0);

    // plane shape (N,C,h_in,w_in)
    const int n = plane[0].size(0); // n == 1
    const int c = plane[0].size(1); // c == 16
    const int block_in = plane[0].size(2); // block_in == 8

    // plane_line_ptr的赋值在此进行:
    int arraySize = plane.size();    // tensorf.density_plane的长度,只测试过为3的情况, 待补充为2情况的测试
    float* plane_line_ptr_cpu[arraySize * 2];       // 各个plane和line平面数据的指针
    // 赋值参数数组, 在cpu端
    for(int idx = 0; idx < plane.size(); idx++){
      plane_line_ptr_cpu[idx * 2] = plane[idx].data<float>();
      plane_line_ptr_cpu[idx * 2 + 1] = line[idx].data<float>();
    }
    // 拷贝到GPU端
    float** plane_line_ptr_gpu = (float**)plane_line_ptr.data<int>();
    cudaMemcpy(plane_line_ptr_gpu, plane_line_ptr_cpu, arraySize * sizeof(float*) * 2, cudaMemcpyHostToDevice);

    const dim3 block(num_block, 1, 1);
    const dim3 grid(int((xyz_num - 1) / num_block) + 1, 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_sum_xyz_b_3_cuda_noMalloc", ([&] {
    fusedkernel_sum_xyz_b_3<<<grid, block>>>(
        xyz.data<float>(),
        b.data<int>(),
        plane_line_ptr_gpu,
        sigma.data<float>(),
        xyz_num,
        block_in,
        hw_in.data<int>(),
        c, arraySize,
        false, num_block);
    }));
    return 0;
}


int compute_grid_sample_and_sum_xyz_b3_noMalloc_nbhwc_cuda(
  torch::Tensor xyz, torch::Tensor b, std::vector<torch::Tensor> plane, std::vector<torch::Tensor> line,
  torch::Tensor hw_in, torch::Tensor plane_line_ptr,
  torch::Tensor sigma, int num_block, bool judgeOverflow)
{
    const int xyz_num = xyz.size(0);

    // plane shape (N, Block, h_in, w_in, channel)
    const int n = plane[0].size(0); // n == 1
    const int block_in = plane[0].size(1); // block_in == 8
    const int c = plane[0].size(4); // c == 16


    // plane_line_ptr的赋值在此进行:
    int arraySize = plane.size();    // tensorf.density_plane的长度,只测试过为3的情况, 待补充为2情况的测试
    float* plane_line_ptr_cpu[arraySize * 2];       // 各个plane和line平面数据的指针
    // 赋值参数数组, 在cpu端
    for(int idx = 0; idx < plane.size(); idx++){
      plane_line_ptr_cpu[idx * 2] = plane[idx].data<float>();
      plane_line_ptr_cpu[idx * 2 + 1] = line[idx].data<float>();
    }
    // 拷贝到GPU端
    float** plane_line_ptr_gpu = (float**)plane_line_ptr.data<int>();
    cudaMemcpy(plane_line_ptr_gpu, plane_line_ptr_cpu, arraySize * sizeof(float*) * 2, cudaMemcpyHostToDevice);

    const dim3 block(num_block, 1, 1);
    const dim3 grid(int((xyz_num - 1) / num_block) + 1, 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_and_sum_xyz_b_3_cuda_noMalloc_nbhwc", ([&] {
    fusedkernel_sum_xyz_b_3_nbhwc<<<grid, block>>>(
        xyz.data<float>(),
        b.data<int>(),
        plane_line_ptr_gpu,
        sigma.data<float>(),
        xyz_num,
        block_in,
        hw_in.data<int>(),
        c, arraySize,
        judgeOverflow, num_block);
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

int GridSample2D_nhwc_shared_cuda(torch::Tensor xyz, torch::Tensor plane, torch::Tensor sigma)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);

    // plane shape (N,h_in,w_in,C)  [1, 2744, 2744, 48]
    const int n = plane.size(0); // n == 1
    const int h_in = plane.size(1); // h_in == 2047
    const int w_in = plane.size(2); // w_in == 2047
    const int c = plane.size(3); // c == 48
    const dim3 block(c, NUM_BLOCK_64, 1);
    const dim3 grid(int((xyz_num - 1) / NUM_BLOCK_64) + 1, 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_samplet_test_nhwc", ([&] {
        kernel2d_nhwc_shared<scalar_t><<<grid, block>>>(
            xyz.data<scalar_t>(),
            plane.data<scalar_t>(),
            sigma.data<scalar_t>(),
            xyz_num,
            c,
            h_in,
            w_in);
        }));
    return 0;
}
