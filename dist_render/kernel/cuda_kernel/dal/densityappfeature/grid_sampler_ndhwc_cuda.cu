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

// At the kernel level, fusion is performed, where one thread computes the sum of three planes directly. The calculation uses nbhwc as input, and the output is [c].
__global__ void fusedkernel_sum_xyz_b_3_nbhwc
(const float* xyz, const int* b, float** plane_line_ptr_not_share, float* sigma,
  const int xyz_num, const int block_in, const int* hw_in_not_share, const int C, const int arraySize, bool judgeOverflow, int num_block) {
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
    // There are num_block threads per SM, and each thread uses 4 float4 spaces.
    __shared__ float plane_channel[10000];

  if (((int)threadIdx.x < (min(((- num_block  * (int)blockIdx.x) + (xyz_num-1)), (num_block - 1)) + 1))) {
    float sum;
    sum = 0.0;
    unsigned int xyzlen = 3; //4
    int ib = b[((int)blockIdx.x * num_block) + (int)threadIdx.x];
    float ix, iy, iz;
    float result_plane, result_line;
    int m_id = (((int)blockIdx.x * num_block) + (int)threadIdx.x);

    for (int planeid = 0; planeid < arraySize; planeid++){
      // hw_in: h_in, w_in, line_h_in, line_w_in
      const int h_in = hw_in[planeid * 4 + 0];
      const int w_in = hw_in[planeid * 4 + 1];
      const int line_h_in = hw_in[planeid * 4 + 2];
      const int line_w_in = hw_in[planeid * 4 + 3];

      float * plane = plane_line_ptr[planeid * 2 + 0];
      float4* plane4 = reinterpret_cast<float4*>(plane);
      const float * line = plane_line_ptr[planeid * 2 + 1];

                             // IDX2C(xyz_id, m_pos, xyzlen), m_pos = (((int)blockIdx.x * num_block) + (int)threadIdx.x)
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


__global__ void fusedkernel_serial_xyb_bz_nbhwc2_outhalf(const float* xyz, const int* b, float* plane, const float* line, half* sigma,
const int xyz_num, const int block_in, const int h_in, const int w_in,
const int line_h_in, const int line_w_in, const int C, const int nanobatch, const int plane_idx, const int plane_in) {
  __shared__ float sample_grid_shared[8 * 8 * 10];  // One block contains 8 job groups, each group has 8 valid points, and each valid point requires 10 float spaces for storage.

  float *sample_grid_thisjob = sample_grid_shared + threadIdx.y * 80;

  int job_id = blockIdx.x * blockDim.y + threadIdx.y;  // blockDim.y == job_num_per_block
  int point_pos_base = job_id * nanobatch;

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


    int nanobatch = serial;                 // A cluster of threadx performs calculations for nanobatch valid points, where each threadidx is responsible for computing the corresponding channel. We consider this cluster of calculations as one job.
    int job_num_per_block = 8;              // There are a total of m / nano jobs.

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



    int arraySize = plane.size();
    float* plane_line_ptr_cpu[arraySize * 2];

    for(int idx = 0; idx < plane.size(); idx++){
      plane_line_ptr_cpu[idx * 2] = plane[idx].data<float>();
      plane_line_ptr_cpu[idx * 2 + 1] = line[idx].data<float>();
    }

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
