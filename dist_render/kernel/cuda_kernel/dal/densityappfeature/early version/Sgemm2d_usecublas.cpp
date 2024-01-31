#include <torch/extension.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

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
    const int M = Ga.size(0);
    const int K = Ga.size(1);
    const int N = Gb.size(1);
    // 实现矩阵乘法，C=A*B


    constexpr const int TP = 16;
    dim3 threadsPer(TP, TP);
    dim3 blocksPer((M + TP - 1) / TP, (N + TP - 1) / TP);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1, beta = 0;

    //C=alpha*A*B+beta,
    //cublas中矩阵是列优先的格式，而C++是行优先的格式,所以调用的时候是d_B在前，d_A在后 C^T = B^T*A^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, Gb.data<float>(), N, Ga.data<float>(), K, &beta, Gc.data<float>(), N);
    return;
    //返回result,result也是py::array_t<T>格式，也就是python中 的numpy.ndarray
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{.
    m.def("Gpu_Cublas", &ts_multiply_Cublas, "Multuply tow arrays use cublas");
    //"Gpu_Cublas"代表python中对应的函数，&np_multiply_Cublas是对应的C++函数指针，之后的字符串是python中的函数doc
}
