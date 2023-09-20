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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

// CUDA funciton declearition
int compute_grid_sample_and_ewproduct_cuda(
    torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void ts_multiply_Cublas(torch::Tensor Ga, torch::Tensor Gb, torch::Tensor Gc)
// (py::array_t<float> &inLeft, py::array_t<float> &inRight)
{
    // CHECK_INPUT(Ga);
    // CHECK_INPUT(Gb);
    // CHECK_INPUT(Gc);
    //request方法活得对py::array_t<T>的绑定，包括维度、数据指针、size、shape等参数
    assert(Ga.size(1) == Gb.size(0));
    //M、K、N分别是A的行数、列数、B的列数，C的shape为{M，N}
    const int M = Ga.size(0); //std::cout << " M = \t"  << M << "\n";
    const int K = Ga.size(1); //std::cout << " K = \t"  << K << "\n";
    const int N = Gb.size(1); //std::cout << " N = \t"  << N << "\n";
    // 实现矩阵乘法，C=A*B


    constexpr const int TP = 16;
    dim3 threadsPer(TP, TP);
    dim3 blocksPer((M + TP - 1) / TP, (N + TP - 1) / TP);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1, beta = 0;

    //C=alpha*A*B+beta,
    //cublas中矩阵是列优先的格式，而C++是行优先的格式,所以调用的时候是d_B在前，d_A在后 C^T = B^T*A^T
    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha,
        Ga.data<float>(), K,
        Gb.data<float>(), N,
        &beta, Gc.data<float>(), M);
    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, Gb.data<float>(), N, Ga.data<float>(), K, &beta, Gc.data<float>(), N);
    return;
    //返回result,result也是py::array_t<T>格式，也就是python中 的numpy.ndarray
}

// for app_feature, the fourth parameter: sigma correspond to app_features(237684*48)
void compute_grid_sample_and_ewproduct(
    torch::Tensor xyz, std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec, torch::Tensor sigma_vec, torch::Tensor Ga, torch::Tensor Gc)
{
    // bool debug = 1;
    // if(debug){
    //     cudaEvent_t start, stop1, stop2, stop3, instart, instop;
    //     cudaEventCreate(&start);
    //     cudaEventCreate(&stop1);
    //     cudaEventCreate(&stop2);
    //     cudaEventCreate(&stop3);
    //     cudaEventCreate(&instart);
    //     cudaEventCreate(&instop);
    //     cudaEventRecord(start);
    // }
    CHECK_INPUT(xyz);
	for(int idx = 0; idx < plane_vec.size(); idx++ ){
        // if(debug){
        //     cudaEventRecord(instart);
        //     CHECK_INPUT(plane_vec[idx]);
        //     CHECK_INPUT(line_vec[idx]);
        //     CHECK_INPUT(sigma_vec[idx]);
        // }
        compute_grid_sample_and_ewproduct_cuda(xyz, plane_vec[idx], line_vec[idx], sigma_vec[idx]);

        // if(debug)
        //     cudaEventRecord(instop);
        //     cudaDeviceSynchronize();
        //     float inTime;
        //     cudaEventElapsedTime(&inTime, instart, instop);
        //     std::cout << "        grid_sample\t"  << idx << ":  " << inTime << " ms\n";
	}
    // if(debug)
    //     cudaEventRecord(stop1);
    // auto sigma_cat = torch::cat({sigma_vec[0], sigma_vec[1], sigma_vec[2]}, 0);
    sigma_vec = sigma_vec.view({-1,xyz.size(0)});
    // if(debug)
    //     cudaEventRecord(stop2);
    ts_multiply_Cublas(Ga, sigma_vec, Gc);
    // if(debug){
    //     cudaEventRecord(stop3);
    //     cudaDeviceSynchronize();
    //     float Time1, Time2, Time3, Timea;
    //     cudaEventElapsedTime(&Time1, start, stop1);
    //     cudaEventElapsedTime(&Time2, stop1, stop2);
    //     cudaEventElapsedTime(&Time3, stop2, stop3);
    //     cudaEventElapsedTime(&Timea, start, stop3);
    //     std::cout << "    grid_sample \t"  << Time1<< " ms\n";
    //     std::cout << "    cat time \t" << Time2 << " ms\n";
    //     std::cout << "    cublas time \t" << Time3 << " ms\n";
    //     std::cout << "    all time in C-level " << Timea  << " ms\n";
    // }


    return;


}
// void compute_grid_sample_and_ewproduct2(
//     torch::Tensor xyz, std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec, std::vector<torch::Tensor> sigma_vec, torch::Tensor Ga, torch::Tensor Gc)
// {

//     cudaEvent_t start, stop1, stop2, stop3, instart, instop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop1);
//     cudaEventCreate(&stop2);
//     cudaEventCreate(&stop3);
//     cudaEventCreate(&instart);
//     cudaEventCreate(&instop);
//     cudaEventRecord(start);
//     CHECK_INPUT(xyz);
// 	for(int idx = 0; idx < plane_vec.size(); idx++ ){
//         cudaEventRecord(instart);
//         CHECK_INPUT(plane_vec[idx]);
//         CHECK_INPUT(line_vec[idx]);
//         CHECK_INPUT(sigma_vec[idx]);
//         compute_grid_sample_and_ewproduct_cuda(xyz, plane_vec[idx], line_vec[idx], sigma_vec[idx]);
//         sigma_vec[idx] = sigma_vec[idx].view({-1,xyz.size(0)});
//         cudaEventRecord(instop);
//         cudaDeviceSynchronize();
//         float inTime;
//         cudaEventElapsedTime(&inTime, instart, instop);
//         std::cout << "        grid_sample\t"  << idx << ":  " << inTime << " ms\n";
// 	}

//     cudaEventRecord(stop1);

//     auto sigma_cat = torch::cat({sigma_vec[0], sigma_vec[1], sigma_vec[2]}, 0);

//     // auto sigma_cat = torch::cat({sigma_vec[0], sigma_vec[1]}, 0);


//     // sigma_vec = sigma_vec.view({-1,xyz.size(0)});
//     cudaEventRecord(stop2);
//     ts_multiply_Cublas(Ga, sigma_cat, Gc);
//     cudaEventRecord(stop3);


//     cudaDeviceSynchronize();
//     float Time1, Time2, Time3, Timea;
//     cudaEventElapsedTime(&Time1, start, stop1);
//     cudaEventElapsedTime(&Time2, stop1, stop2);
//     cudaEventElapsedTime(&Time3, stop2, stop3);
//     cudaEventElapsedTime(&Timea, start, stop3);
//     std::cout << "    grid_sample \t"  << Time1<< " ms\n";
//     std::cout << "    cat time \t" << Time2 << " ms\n";
//     std::cout << "    cublas time \t" << Time3 << " ms\n";
//     std::cout << "    all time in C-level " << Timea  << " ms\n";

//     return;


// }



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda", &compute_grid_sample_and_ewproduct, "compute_grid_sample_and_ewproduct (CUDA)");
//   m.def("cuda2", &compute_grid_sample_and_ewproduct2, "compute_grid_sample_and_ewproduct (CUDA)");
}
