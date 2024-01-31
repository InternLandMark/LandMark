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
// SOFTWAREalpha

#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


#include <unistd.h>
#include <stdlib.h>
#include <chrono>
using namespace std;
using namespace chrono;

#include <time.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int compute_grid_sample_and_ewproduct_serial_xyb_bz_nbhwc2_outhalf_cuda(
    torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial, int plane_idx, int plane_in);

int compute_grid_sample_and_sum_xyz_b3_noMalloc_nbhwc_cuda(
    torch::Tensor xyz, torch::Tensor b, std::vector<torch::Tensor> plane, std::vector<torch::Tensor> line,
    torch::Tensor hw_in,
    torch::Tensor plane_line_ptr,
    torch::Tensor sigma, int num_block, bool judgeOverflow);


void compute_gridsample_and_ewproduct_xyb_bz_nbhwc2(
    torch::Tensor xyz, torch::Tensor b, std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec, torch::Tensor sigma_vec, int serial = 1, int debug = 0)
{
    CHECK_INPUT(xyz);
    CHECK_INPUT(b);
    cudaEvent_t start, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    if (debug == 1){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
    }


    torch::ScalarType dtype_in =plane_vec[0].scalar_type();
    torch::ScalarType dtype_out =sigma_vec[0].scalar_type();

    for(int idx = 0; idx < plane_vec.size(); idx++ ){
        CHECK_INPUT(plane_vec[idx]);
        CHECK_INPUT(line_vec[idx]);
        CHECK_INPUT(sigma_vec[idx]);

        if (dtype_out == torch::ScalarType::Float){
            std::cout << "    kernel do not support dtype_out == torch::ScalarType::float\n";
        }else if(dtype_out == torch::ScalarType::Half){
            compute_grid_sample_and_ewproduct_serial_xyb_bz_nbhwc2_outhalf_cuda(xyz, b, plane_vec[idx], line_vec[idx], sigma_vec, serial, idx, plane_vec.size());
        }
    }

    if (debug == 1){
        cudaEventRecord(stop1);
        cudaDeviceSynchronize();
        float Time1;
        cudaEventElapsedTime(&Time1, start, stop1);
        std::cout << "    compute_gridsample_and_ewproduct_xyb_bz_nbhwc \t"  << Time1<< " ms\n";
    }
    return;
}

void compute_gridsample_and_sum_xyz_b3_noMalloc_nbhwc(torch::Tensor xyz, torch::Tensor b, std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec,
torch::Tensor hw_in,
torch::Tensor plane_line_ptr,
torch::Tensor sigma, int debug, int num_block, bool judgeOverflow)
{
    // CHECK_INPUT(xyz);
    // CHECK_INPUT(plane);
    // CHECK_INPUT(line);
    // CHECK_INPUT(sigma);

    // std::cout << "here1 "<< "\n";
    // std::cout << b[0] << "\n";
    // std::cout << h_in << "\n";

    if(debug == 0){
        compute_grid_sample_and_sum_xyz_b3_noMalloc_nbhwc_cuda(xyz, b, plane_vec, line_vec, hw_in, plane_line_ptr, sigma, num_block, judgeOverflow);
    }else{
        cudaEvent_t start, stop1;
        cudaEventCreate(&start);
        cudaEventCreate(&stop1);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        compute_grid_sample_and_sum_xyz_b3_noMalloc_nbhwc_cuda(xyz, b, plane_vec, line_vec, hw_in, plane_line_ptr, sigma, num_block, judgeOverflow);
        cudaEventRecord(stop1);
        cudaDeviceSynchronize();

        float Time1;
        cudaEventElapsedTime(&Time1, start, stop1);
        std::cout << "    compute_grid_sample_and_sum_xyz_b3_noMalloc_nbhwc_cuda \t"  << Time1<< " ms\n";
    }
    return;
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gridsample_ew_xyb_bz_nbhwc2", &compute_gridsample_and_ewproduct_xyb_bz_nbhwc2);
  m.def("gridsample_sum_xyz_b3_noMalloc_nbhwc", &compute_gridsample_and_sum_xyz_b3_noMalloc_nbhwc);
}
