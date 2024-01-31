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

#define NUM_BLOCK 512

inline __host__ __device__ float runtime_exp(float x) { return expf(x); }
inline __host__ __device__ double runtime_exp(double x) { return exp(x); }

// template <typename scalar_t>
// __global__ void  kernel0(const scalar_t* sigma, const scalar_t* dist, scalar_t* alpha, scalar_t* beta, const int elements_num) {
//     if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (elements_num-1)), (NUM_BLOCK-1)) + 1))) {
//         beta[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = runtime_exp(((0 - sigma[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]) * dist[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]));
//         alpha[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = (1 - beta[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]);
//     }
// }

template <typename scalar_t>
__global__ void  kernel0(const scalar_t* sigma, const scalar_t* dist, scalar_t* beta, const int elements_num) {
    if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (elements_num-1)), (NUM_BLOCK-1)) + 1))) {
        beta[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = runtime_exp(((0 - sigma[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]) * dist[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]));
    }
}

template <typename scalar_t>
__global__ void  computebeta_kernel1(const scalar_t* sigma, const scalar_t* z_val, scalar_t* beta, float distance_scale, const int n_samples, const int elements_num) {
    if (((int)threadIdx.x < (min((
        (-NUM_BLOCK * (int)blockIdx.x) + (elements_num-1)), (NUM_BLOCK-1)) + 1)
        )) {
        scalar_t dist = 0.0;
        if ( (((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) % n_samples != (n_samples-1) ) {
            dist = (z_val[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x + 1) ]
            - z_val[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]) * distance_scale;
        }
        beta[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = runtime_exp(
            (
                (0 - sigma[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)])
                 * dist
            ));
    }
}


std::vector<torch::Tensor> compute_beta_cuda(
    // torch::Tensor sigma, torch::Tensor dist, torch::Tensor beta)
    torch::Tensor sigma, torch::Tensor z_val, torch::Tensor beta, float distance_scale)
{
    const int elements = sigma.size(0) * sigma.size(1); // only support 2D condition
    const int n_samples = sigma.size(1);

    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid((elements - 1) / NUM_BLOCK + 1, 1, 1);

    // AT_DISPATCH_FLOATING_TYPES(sigma.type(), "compute_beta_cuda", ([&] {
    //     kernel0<scalar_t><<<grid, block>>>(
    //         sigma.data<scalar_t>(),
    //         dist.data<scalar_t>(),
    //         // alpha.data<scalar_t>(),
    //         beta.data<scalar_t>(),
    //         elements);
    //     }));

    AT_DISPATCH_FLOATING_TYPES(sigma.type(), "compute_beta_cuda", ([&] {
        computebeta_kernel1<scalar_t><<<grid, block>>>(
            sigma.data<scalar_t>(),
            z_val.data<scalar_t>(),
            beta.data<scalar_t>(),
            distance_scale,
            n_samples,
            elements);
        }));

    return {beta};
}
