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


__global__ void fusekernel0(const float* xyz, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C) {
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
    float sum;
    sum = 0.0;
    for (int cc = 0; cc < C; cc++) {
      float ix;
      float iy;
      int32_t ix_nw;
      int32_t iy_nw;
      float result_plane;
      ix = ((float(w_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
      iy = ((float(h_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
      ix_nw = int32_t(ix);
      iy_nw = int32_t(iy);

      result_plane = (((((plane[cc * h_in * w_in + iy_nw * w_in + ix_nw] * (1.0 - (iy - float(iy_nw)))) * (1.0  - (ix - float(ix_nw)))) + ((plane[cc * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] * (iy - float(iy_nw))) * (1.0  - (ix - float(ix_nw))))) + ((plane[cc * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] * (1.0  - (iy - float(iy_nw)))) * (ix - float(ix_nw)))) + ((plane[cc * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] * (iy - float(iy_nw))) * (ix - float(ix_nw))));

      ix = ((float(line_w_in-1) * (1.0)) / 2.0);
      iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
      ix_nw = int32_t(ix);
      iy_nw = int32_t(iy);
      float result_line;

      result_line = (((((line[cc * line_h_in * line_w_in + iy_nw * line_w_in + ix_nw] * (1.0 - (iy - float(iy_nw)))) * (1.0  - (ix - float(ix_nw)))) + ((line[cc * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + ix_nw] * (iy - float(iy_nw))) * (1.0  - (ix - float(ix_nw))))) + ((line[cc * line_h_in * line_w_in + iy_nw * line_w_in + min((ix_nw + 1), (line_w_in-1))] * (1.0  - (iy - float(iy_nw)))) * (ix - float(ix_nw)))) + ((line[cc * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + min((ix_nw + 1), (line_w_in-1))] * (iy - float(iy_nw))) * (ix - float(ix_nw))));

      sum += result_plane * result_line;
    }
    sigma[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = sum;
  }
}


__global__ void fusedkernel1(const float* xyz, const float* plane, const float* line, float* sigma, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in, const int C) {
  // printf("threadIdx.x: %d \t min: %d \t  is_true?%d\n", threadIdx.x, (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1), (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))));
  if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
    float ix;
    float iy;
    int32_t ix_nw;
    int32_t iy_nw;
    ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
    iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
    ix_nw = int32_t(ix);
    iy_nw = int32_t(iy);
    float plane_coef_tmp;
    plane_coef_tmp = (((((plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + ix_nw] * (1.0 - (iy - float(iy_nw)))) * (1.0  - (ix - float(ix_nw)))) + ((plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] * (iy - float(iy_nw))) * (1.0  - (ix - float(ix_nw))))) + ((plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] * (1.0  - (iy - float(iy_nw)))) * (ix - float(ix_nw)))) + ((plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] * (iy - float(iy_nw))) * (ix - float(ix_nw))));

    ix = ((float(line_w_in-1) * (1.0)) / 2.0);
    iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
    ix_nw = int32_t(ix);
    iy_nw = int32_t(iy);

    float line_coef_tmp;
    line_coef_tmp = (((((line[(int)blockIdx.x * line_h_in * line_w_in + iy_nw * line_w_in + ix_nw] * (1.0 - (iy - float(iy_nw)))) * (1.0  - (ix - float(ix_nw)))) + ((line[(int)blockIdx.x * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + ix_nw] * (iy - float(iy_nw))) * (1.0  - (ix - float(ix_nw))))) + ((line[(int)blockIdx.x * line_h_in * line_w_in + iy_nw * line_w_in + min((ix_nw + 1), (line_w_in-1))] * (1.0  - (iy - float(iy_nw)))) * (ix - float(ix_nw)))) + ((line[(int)blockIdx.x * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + min((ix_nw + 1), (line_w_in-1))] * (iy - float(iy_nw))) * (ix - float(ix_nw))));
    sigma[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = plane_coef_tmp * line_coef_tmp;
    // float w = ix - float(ix_nw);
    // float e = 1.0 - w;
    // float n = iy - float(iy_nw);
    // float s = 1.0 - n;

    // float left_top = plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + ix_nw]  * s * e;
    // float left_bottom = plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] * n * e;
    // float right_top = plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] * s * w;
    // float right_bottom = plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] * n * w;

    // float result = left_bottom + left_top + right_bottom + right_top;

    // plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = result;
  }
}
__global__ void kernel0(const float* xyz, const float* plane, float* plane_coef, const float* line, float* line_coef, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in) {
    if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
        float ix;
        float iy;
        int32_t ix_nw;
        int32_t iy_nw;
        ix = ((float(w_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 0] + 1.0)) / 2.0);
        iy = ((float(h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 1] + 1.0)) / 2.0);
        ix_nw = int32_t(ix);
        iy_nw = int32_t(iy);

        plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = (((((plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + ix_nw] * (1.0 - (iy - float(iy_nw)))) * (1.0  - (ix - float(ix_nw)))) + ((plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] * (iy - float(iy_nw))) * (1.0  - (ix - float(ix_nw))))) + ((plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] * (1.0  - (iy - float(iy_nw)))) * (ix - float(ix_nw)))) + ((plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] * (iy - float(iy_nw))) * (ix - float(ix_nw))));

        ix = ((float(line_w_in-1) * (1.0)) / 2.0);
        iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
        ix_nw = int32_t(ix);
        iy_nw = int32_t(iy);

        line_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = (((((line[(int)blockIdx.x * line_h_in * line_w_in + iy_nw * line_w_in + ix_nw] * (1.0 - (iy - float(iy_nw)))) * (1.0  - (ix - float(ix_nw)))) + ((line[(int)blockIdx.x * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + ix_nw] * (iy - float(iy_nw))) * (1.0  - (ix - float(ix_nw))))) + ((line[(int)blockIdx.x * line_h_in * line_w_in + iy_nw * line_w_in + min((ix_nw + 1), (line_w_in-1))] * (1.0  - (iy - float(iy_nw)))) * (ix - float(ix_nw)))) + ((line[(int)blockIdx.x * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + min((ix_nw + 1), (line_w_in-1))] * (iy - float(iy_nw))) * (ix - float(ix_nw))));

        // float w = ix - float(ix_nw);
        // float e = 1.0 - w;
        // float n = iy - float(iy_nw);
        // float s = 1.0 - n;

        // float left_top = plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + ix_nw]  * s * e;
        // float left_bottom = plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] * n * e;
        // float right_top = plane[(int)blockIdx.x * h_in * w_in + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] * s * w;
        // float right_bottom = plane[(int)blockIdx.x * h_in * w_in + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] * n * w;

        // float result = left_bottom + left_top + right_bottom + right_top;

        // plane_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = result;
      }
  }

//   __global__ void kernel1(const float* xyz, const float* plane, float* plane_coef, const float* line, float* line_coef, const int xyz_num, const int h_in, const int w_in, const int line_h_in, const int line_w_in) {
//     if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.y) + (xyz_num-1)), (NUM_BLOCK-1)) + 1))) {
//         float ix;
//         float iy;
//         int32_t ix_nw;
//         int32_t iy_nw;

//         ix = ((float(line_w_in-1) * (1.0)) / 2.0);
//         iy = ((float(line_h_in-1) * (xyz[(((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x) * 3 + 2] + 1.0)) / 2.0);
//         ix_nw = int32_t(ix);
//         iy_nw = int32_t(iy);

//         line_coef[(int)blockIdx.x * xyz_num + (((int)blockIdx.y * NUM_BLOCK) + (int)threadIdx.x)] = (((((line[(int)blockIdx.x * line_h_in * line_w_in + iy_nw * line_w_in + ix_nw] * (1.0 - (iy - float(iy_nw)))) * (1.0  - (ix - float(ix_nw)))) + ((line[(int)blockIdx.x * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + ix_nw] * (iy - float(iy_nw))) * (1.0  - (ix - float(ix_nw))))) + ((line[(int)blockIdx.x * line_h_in * line_w_in + iy_nw * line_w_in + min((ix_nw + 1), (line_w_in-1))] * (1.0  - (iy - float(iy_nw)))) * (ix - float(ix_nw)))) + ((line[(int)blockIdx.x * line_h_in * line_w_in + min((iy_nw + 1), (line_h_in-1)) * line_w_in + min((ix_nw + 1), (line_w_in-1))] * (iy - float(iy_nw))) * (ix - float(ix_nw))));
//       }
//   }

// std::vector<torch::Tensor> compute_grid_sample_cuda(
//     torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma)
// {
//     // xyz shape (xyz_num, 3)
//     const int xyz_num = xyz.size(0); //xyz_num == 237684
//     // assert(xyz.size(1)==3);

//     // plane shape (N,C,h_in,w_in)
//     const int n = plane.size(0); // n == 1
//     // assert(n==1);
//     const int c = plane.size(1); // c == 48
//     const int h_in = plane.size(2); // h_in == 2047
//     const int w_in = plane.size(3); // w_in == 2047

//     // line shape (N,C,line_h_in,line_w_in)
//     // assert(line.size(0) == n);
//     // assert(line.size(1) == c);
//     const int line_h_in = line.size(2); // line_h_in == 255
//     const int line_w_in = line.size(3); // line_w_in == 1

//     // const dim3 block(NUM_BLOCK, 1, 1);
//     // const dim3 grid(c, (xyz_num - 1) / NUM_BLOCK + 1, 1);

//     // AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_cuda", ([&] {
//     //     kernel0<<<grid, block>>>(
//     //         xyz.data<float>(),
//     //         plane.data<float>(),
//     //         plane_coef.data<float>(),
//     //         line.data<float>(),
//     //         line_coef.data<float>(),
//     //         xyz_num,
//     //         h_in,
//     //         w_in,
//     //         line_h_in,
//     //         line_w_in);
//     //     }));

//     // return {plane_coef, line_coef};
//     const dim3 block(NUM_BLOCK, 1, 1);
//     const dim3 grid(int((xyz_num - 1) / NUM_BLOCK) + 1, 1, 1);

//     AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_cuda", ([&] {
//         fusekernel0<<<grid, block>>>(
//             xyz.data<float>(),
//             plane.data<float>(),
//             line.data<float>(),
//             sigma.data<float>(),
//             xyz_num,
//             h_in,
//             w_in,
//             line_h_in,
//             line_w_in,
//             c);
//         }));

//     return {sigma};
// }


int compute_grid_sample_and_ewproduct_cuda(torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma)
{
    // xyz shape (xyz_num, 3)
    const int xyz_num = xyz.size(0);
    // assert(xyz.size(1)==3);

    // plane shape (N,C,h_in,w_in)
    const int n = plane.size(0); // n == 1
    // assert(n==1);
    const int c = plane.size(1); // c == 48
    const int h_in = plane.size(2); // h_in == 2047

    const int w_in = plane.size(3); // w_in == 2047

    // line shape (N,C,line_h_in,line_w_in)
    // assert(line.size(0) == n);
    // assert(line.size(1) == c);
    const int line_h_in = line.size(2); // line_h_in == 255
    const int line_w_in = line.size(3); // line_w_in == 1
    // const dim3 block(NUM_BLOCK, 1, 1);
    // const dim3 grid(c, (xyz_num - 1) / NUM_BLOCK + 1, 1);

    // AT_DISPATCH_FLOATING_TYPES(xyz.type(), "compute_grid_sample_cuda", ([&] {
    //     kernel0<<<grid, block>>>(
    //         xyz.data<float>(),
    //         plane.data<float>(),
    //         plane_coef.data<float>(),
    //         line.data<float>(),
    //         line_coef.data<float>(),
    //         xyz_num,
    //         h_in,
    //         w_in,
    //         line_h_in,
    //         line_w_in);
    //     }));

    // return {plane_coef, line_coef};
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
