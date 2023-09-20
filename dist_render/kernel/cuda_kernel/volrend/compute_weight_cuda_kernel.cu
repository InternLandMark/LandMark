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

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define NUM_BLOCK 128
#define THRESHOLD 120

__device__ float WarpScan(float val) {
    int lane = threadIdx.x & 31;
    float tmp = __shfl_up_sync(0xffffffff, val, 1);
    if (lane >= 1) val *= tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 2);
    if (lane >= 2) val *= tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 4);
    if (lane >= 4) val *= tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 8);
    if (lane >= 8) val *= tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 16);
    if (lane >= 16) val *= tmp;
    __syncthreads();
    return val;
}

__device__ float BlockScan(float val, const int n_samples) {
    int warp_id = threadIdx.x >> 5; // divide 32
    int lane = threadIdx.x & 31;
    __shared__ float warp_sum[32];


    val = WarpScan(val);
    __syncthreads();

    if(lane == 31) warp_sum[warp_id] = val;

    __syncthreads();
    if(warp_id == 0) {
        if (lane >= 1) warp_sum[lane] *= warp_sum[lane-1];
        __syncwarp();
        if (lane >= 2) warp_sum[lane] *= warp_sum[lane-2];
        __syncwarp();
        if (lane >= 4) warp_sum[lane] *= warp_sum[lane-4];
        __syncwarp();
        if (lane >= 8) warp_sum[lane] *= warp_sum[lane-8];
        __syncwarp();
        if (lane >= 16) warp_sum[lane] *= warp_sum[lane-16];
    }
    __syncthreads();
    if(warp_id > 0) val *= warp_sum[warp_id-1];
    __syncthreads();
    return val;
}

__global__ void ScanKernel(const float* beta, float* weight, const int n_rays, const int n_samples) {
    int block_id = blockIdx.x;
    int t_id = threadIdx.x;
    int idx = block_id * n_samples + t_id;
    int next_idx = block_id * n_samples + t_id + 1;

    float val = 1.0;
    if (t_id < n_samples) val = beta[idx] + 1e-10;

    val = BlockScan(val, n_samples);

    if(t_id == 0)
        weight[idx] = 1 - beta[idx];
    if(t_id < n_samples-1)
        weight[next_idx] = (1 - beta[next_idx]) * val;

}

template <typename scalar_t>
__global__ void  kernel0(const scalar_t* beta, scalar_t* weight, const int n_rays, const int n_samples) {
    if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (n_rays-1)), (NUM_BLOCK-1)) + 1))) {
        scalar_t cumprod;
        cumprod = 1.0;
        weight[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * n_samples + 0] = ((1 - beta[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * n_samples +  0]) * cumprod);
        for (int j = 0; j < (n_samples-1); j++) {
            cumprod *= beta[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * n_samples + j];
            weight[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)*  n_samples + (j + 1)] = ((1 - beta[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * n_samples + (j + 1)]) * cumprod);
        }
}
}

std::vector<torch::Tensor> compute_weight_cuda(
    torch::Tensor beta, torch::Tensor weight)
{
    const int n_rays = beta.size(0); // only support 2D condition
    const int n_samples = beta.size(1);

    if (n_samples >= THRESHOLD) {
        const dim3 block( (int)((n_samples+32-1)/32) * 32, 1, 1);
        const dim3 grid(n_rays, 1, 1);

        AT_DISPATCH_FLOATING_TYPES(beta.type(), "compute_weight_cuda", ([&] {
            ScanKernel<<<grid, block>>>(
                beta.data<float>(),
                weight.data<float>(),
                n_rays,
                n_samples);
            }));
    } else {
        const dim3 block(NUM_BLOCK, 1, 1);
        const dim3 grid((n_rays - 1) / NUM_BLOCK + 1, 1, 1);

        AT_DISPATCH_FLOATING_TYPES(beta.type(), "compute_weight_cuda", ([&] {
            kernel0<scalar_t><<<grid, block>>>(
                beta.data<scalar_t>(),
                weight.data<scalar_t>(),
                n_rays,
                n_samples);
            }));
    }


    return {weight};
}
