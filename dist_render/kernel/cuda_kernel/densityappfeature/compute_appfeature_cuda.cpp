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
// 6个kernel接口：
// 计算appfeature的基础版本，单次的2d+1d的gridsample并按位相乘
int compute_grid_sample_and_ewproduct_cuda(
    torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma);
// 加速方案1：合并两个基础版本
int compute_grid_sample_and_ewproduct_cuda2(
    torch::Tensor xyz, torch::Tensor plane0, torch::Tensor line0, torch::Tensor sigma0,
                       torch::Tensor plane1, torch::Tensor line1, torch::Tensor sigma1);
// 加速方案2：对基础版本做串行化
int compute_grid_sample_and_ewproduct_serial_cuda(
    torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial);
int compute_grid_sample_and_ewproduct_serial_xyz_b_cuda(
    torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial);
int compute_grid_sample_and_ewproduct_serial_xyb_bz_cuda(
    torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial);
int compute_grid_sample_and_ewproduct_serial_xyb_bz_outhalf_cuda(
    torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial);

int compute_grid_sample_and_ewproduct_serial_xyb_bz_half_cuda(
    torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma, int serial);


//  计算density的版本，单次计算两个的gridsample按位相乘后求和,
//  若输入xyz则进行2d+1d, 若输入tensor为xyzb则进行2d+2d
int compute_grid_sample_and_sum_cuda(torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma);
int compute_grid_sample_and_sum_xyz_b_cuda(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma);
//  无误差的1d版本
int GridSample1D_cuda(torch::Tensor xyz, torch::Tensor plane, torch::Tensor sigma);
//  有误差的2d版本
int GridSample2D_cuda(torch::Tensor xyz, torch::Tensor plane, torch::Tensor sigma);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void ts_multiply_Cublas(torch::Tensor Ga, torch::Tensor Gb, torch::Tensor Gc, cublasHandle_t handle)
// (py::array_t<float> &inLeft, py::array_t<float> &inRight)
{
    // CHECK_INPUT(Ga);
    // CHECK_INPUT(Gb);
    // CHECK_INPUT(Gc);
    //request方法活得对py::array_t<T>的绑定，包括维度、数据指针、size、shape等参数

    assert(Ga.size(1) == Gb.size(0));
    //M、K、N分别是A的行数、列数、B的列数，C的shape为{M，N}
    const int M = Ga.size(0);
    const int K = Ga.size(1);
    const int N = Gb.size(1);
    // 实现矩阵乘法，C=A*B


    constexpr const int TP = 16;
    dim3 threadsPer(TP, TP);
    dim3 blocksPer((M + TP - 1) / TP, (N + TP - 1) / TP);

    // cudaEvent_t start, stop1, stop2;
    // cudaEventCreate(&start);
    // cudaEventRecord(start);

    // cublasHandle_t handle;
    // cublasCreate(&handle);

    // cudaEventRecord(stop1);
    float alpha = 1, beta = 0;
    //C=alpha*A*B+beta,
    //cublas中矩阵是列优先的格式，而C++是行优先的格式,所以调用的时候是d_B在前，d_A在后 C^T = B^T*A^T
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, Ga.data<float>(), K, Gb.data<float>(), N, &beta, Gc.data<float>(), M);

    // cudaEventCreate(&stop2);
    // float Time1, Time2;
    // cudaEventElapsedTime(&Time1, start, stop1);
    // cudaEventElapsedTime(&Time2, start, stop2);
    // std::cout << "      cublas ini \t"  << Time1<< " ms\n";
    // std::cout << "      cublasSgemm \t" << Time2 << " ms\n";
    return;
    //返回result,result也是py::array_t<T>格式，也就是python中 的numpy.ndarray
}

// for app_feature, the fourth parameter: sigma correspond to app_features(237684*48)


void compute_appfeature(
    torch::Tensor xyz, std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec, torch::Tensor sigma_vec, torch::Tensor Ga, torch::Tensor Gc)
{
    CHECK_INPUT(xyz);
    cudaEvent_t start, stop1, stop2, stop3, instart, instop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    cudaEventCreate(&stop3);
    cudaEventCreate(&instart);
    cudaEventCreate(&instop);
    cudaEventRecord(start);
    // 固定融合两个grid_sample版本
    if (plane_vec.size() == 2){
    compute_grid_sample_and_ewproduct_cuda2(xyz, plane_vec[0], line_vec[0], sigma_vec[0], plane_vec[1], line_vec[1], sigma_vec[1]);
    }else{
    // 自适应grid_sample版本：
        for(int idx = 0; idx < plane_vec.size(); idx++ ){
            CHECK_INPUT(plane_vec[idx]);
            CHECK_INPUT(line_vec[idx]);
            CHECK_INPUT(sigma_vec[idx]);
            cudaEventRecord(instart);
            compute_grid_sample_and_ewproduct_cuda(xyz, plane_vec[idx], line_vec[idx], sigma_vec[idx]);
            cudaEventRecord(instop);
            cudaDeviceSynchronize();
            float inTime;
            cudaEventElapsedTime(&inTime, instart, instop);
            std::cout << "        grid_sample\t"  << idx << ":  " << inTime << " ms\n";
        }
    }

    cudaEventRecord(stop1);
    // auto sigma_cat = torch::cat({sigma_vec[0], sigma_vec[1], sigma_vec[2]}, 0);
    sigma_vec = sigma_vec.view({-1,xyz.size(0)});

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventRecord(stop2);
    ts_multiply_Cublas(Ga, sigma_vec, Gc, handle);
    cudaEventRecord(stop3);
    cudaDeviceSynchronize();
    float Time1, Time2, Time3, Timea;
    cudaEventElapsedTime(&Time1, start, stop1);
    cudaEventElapsedTime(&Time2, stop1, stop2);
    cudaEventElapsedTime(&Time3, stop2, stop3);
    cudaEventElapsedTime(&Timea, start, stop3);
    std::cout << "    grid_sample \t"  << Time1<< " ms\n";
    std::cout << "    cat time \t" << Time2 << " ms\n";
    std::cout << "    cublas time \t" << Time3 << " ms\n";
    std::cout << "    all time in C-level " << Timea  << " ms\n";
    return;
}

void compute_gridsample_and_ewproduct(
    torch::Tensor xyz, std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec, torch::Tensor sigma_vec, int serial = 1)
{
    CHECK_INPUT(xyz);
    cudaEvent_t start, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventRecord(start);

    for(int idx = 0; idx < plane_vec.size(); idx++ ){
        CHECK_INPUT(plane_vec[idx]);
        CHECK_INPUT(line_vec[idx]);
        CHECK_INPUT(sigma_vec[idx]);
        compute_grid_sample_and_ewproduct_serial_cuda(xyz, plane_vec[idx], line_vec[idx], sigma_vec[idx], serial);
    }
    cudaEventRecord(stop1);
    cudaDeviceSynchronize();
    float Time1;
    cudaEventElapsedTime(&Time1, start, stop1);
    std::cout << "    grid_sample_and_ewproduct \t"  << Time1<< " ms\n";
    return;
}

void compute_gridsample_and_ewproduct_xyz_b(
    torch::Tensor xyz, torch::Tensor b, std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec, torch::Tensor sigma_vec, int serial = 1)
{
    CHECK_INPUT(xyz);
    CHECK_INPUT(b);
    cudaEvent_t start, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventRecord(start);

    for(int idx = 0; idx < plane_vec.size(); idx++ ){
        CHECK_INPUT(plane_vec[idx]);
        CHECK_INPUT(line_vec[idx]);
        CHECK_INPUT(sigma_vec[idx]);
        compute_grid_sample_and_ewproduct_serial_xyz_b_cuda(xyz, b, plane_vec[idx], line_vec[idx], sigma_vec[idx], serial);
    }
    cudaEventRecord(stop1);
    cudaDeviceSynchronize();
    float Time1;
    cudaEventElapsedTime(&Time1, start, stop1);
    std::cout << "    grid_sample_and_ewproduct_xyz_b \t"  << Time1<< " ms\n";
    return;
}

// void compute_gridsample_and_ewproduct_xyb_bz(
//     torch::Tensor xyz, torch::Tensor b, std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec, torch::Tensor sigma_vec, int serial = 1)
// {
//     CHECK_INPUT(xyz);
//     CHECK_INPUT(b);
//     // cudaEvent_t start, stop1;
//     // cudaEventCreate(&start);
//     // cudaEventCreate(&stop1);
//     // cudaEventRecord(start);

//     for(int idx = 0; idx < plane_vec.size(); idx++ ){
//         CHECK_INPUT(plane_vec[idx]);
//         CHECK_INPUT(line_vec[idx]);
//         CHECK_INPUT(sigma_vec[idx]);
//         compute_grid_sample_and_ewproduct_serial_xyb_bz_cuda(xyz, b, plane_vec[idx], line_vec[idx], sigma_vec[idx], serial);
//     }
//     // cudaEventRecord(stop1);
//     // cudaDeviceSynchronize();
//     // float Time1;
//     // cudaEventElapsedTime(&Time1, start, stop1);
//     // std::cout << "    grid_sample_and_ewproduct_xyz_b \t"  << Time1<< " ms\n";
//     return;
// }
void compute_gridsample_and_ewproduct_xyb_bz(
    torch::Tensor xyz, torch::Tensor b, std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec, torch::Tensor sigma_vec, int serial = 1, int debug = 1)
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
    switch (dtype_in) {
        case torch::ScalarType::Float:
            for(int idx = 0; idx < plane_vec.size(); idx++ ){
                CHECK_INPUT(plane_vec[idx]);
                CHECK_INPUT(line_vec[idx]);
                CHECK_INPUT(sigma_vec[idx]);

                if (dtype_out == torch::ScalarType::Float){
                    compute_grid_sample_and_ewproduct_serial_xyb_bz_cuda(xyz, b, plane_vec[idx], line_vec[idx], sigma_vec[idx], serial);
                }else if(dtype_out == torch::ScalarType::Half){
                    compute_grid_sample_and_ewproduct_serial_xyb_bz_outhalf_cuda(xyz, b, plane_vec[idx], line_vec[idx], sigma_vec[idx], serial);
                }
            }
            break;
        case torch::ScalarType::Half:
            for(int idx = 0; idx < plane_vec.size(); idx++ ){
                CHECK_INPUT(plane_vec[idx]);
                CHECK_INPUT(line_vec[idx]);
                CHECK_INPUT(sigma_vec[idx]);
                compute_grid_sample_and_ewproduct_serial_xyb_bz_half_cuda(xyz, b, plane_vec[idx], line_vec[idx], sigma_vec[idx], serial);
            }
            break;
    }
    if (debug == 1){
        cudaEventRecord(stop1);
        cudaDeviceSynchronize();
        float Time1;
        cudaEventElapsedTime(&Time1, start, stop1);
        std::cout << "    grid_sample_and_ewproduct_xyb_bz \t"  << Time1<< " ms\n";
    }
    return;
}


void compute_gridsample_and_sum(torch::Tensor xyz, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma)
{
    CHECK_INPUT(xyz);
    CHECK_INPUT(plane);
    CHECK_INPUT(line);
    CHECK_INPUT(sigma);
    compute_grid_sample_and_sum_cuda(xyz, plane, line, sigma);
    return;
}

void compute_gridsample_and_sum_xyz_b(torch::Tensor xyz, torch::Tensor b, torch::Tensor plane, torch::Tensor line, torch::Tensor sigma)
{
    CHECK_INPUT(xyz);
    CHECK_INPUT(plane);
    CHECK_INPUT(line);
    CHECK_INPUT(sigma);
    compute_grid_sample_and_sum_xyz_b_cuda(xyz, b, plane, line, sigma);
    return;
}

void GridSample2D(
    torch::Tensor xyz, torch::Tensor plane_vec, torch::Tensor sigma_vec)
{
    CHECK_INPUT(xyz);
    CHECK_INPUT(plane_vec);
    CHECK_INPUT(sigma_vec);

    cudaEvent_t start, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    GridSample2D_cuda(xyz, plane_vec, sigma_vec);
    cudaEventRecord(stop1);
    cudaDeviceSynchronize();

    float Time1;
    cudaEventElapsedTime(&Time1, start, stop1);
    std::cout << "    GridSample2D \t"  << Time1<< " ms\n";
    return;
}


void GridSample1D(torch::Tensor xyz, torch::Tensor line, torch::Tensor sigma)
{
    CHECK_INPUT(xyz);
    CHECK_INPUT(line);
    CHECK_INPUT(sigma);

    cudaEvent_t start, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    GridSample1D_cuda(xyz, line, sigma);
    cudaEventRecord(stop1);
    cudaDeviceSynchronize();

    float Time1;
    cudaEventElapsedTime(&Time1, start, stop1);
    std::cout << "    GridSample1D \t"  << Time1<< " ms\n";
    return;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda", &compute_appfeature, "自适应计算appfeaure");
  m.def("gridsample_ew", &compute_gridsample_and_ewproduct, "自适应地做grid_sample_and_ewproduct,支持serial");
  m.def("gridsample_ew_xyz_b", &compute_gridsample_and_ewproduct_xyz_b, "xyz的输入与b分离");
  m.def("gridsample_ew_xyb_bz", &compute_gridsample_and_ewproduct_xyb_bz, "xyb_bz: xyz的输入与b分离");
  m.def("gridsample_sum", &compute_gridsample_and_sum, "自适应地做grid_sample_and_ewproduct_and_sum,支持xyzb");
  m.def("gridsample_sum_xyz_b", &compute_gridsample_and_sum_xyz_b, "xyz_b: 自适应地做grid_sample_and_ewproduct_and_sum,支持xyzb");
  m.def("cuda_1d", &GridSample1D, "无误差的1d版本");
  m.def("cuda_2d", &GridSample2D, "有误差的2d版本");

//   m.def("cuda2", &compute_grid_sample_and_ewproduct2, "compute_grid_sample_and_ewproduct (CUDA)");
}
