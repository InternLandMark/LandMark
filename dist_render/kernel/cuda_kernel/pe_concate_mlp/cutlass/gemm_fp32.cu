#include <torch/extension.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;                        // <- data type of elements in input matrix A
using ElementInputB = float;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

void gemm_fp32_norelu(torch::Tensor Ga, torch::Tensor Gb0,  torch::Tensor Gd0){
     // This code section describes the epilogue part of the kernel
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                     // <- data type of output matrix
        128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                        // memory access. For a byte, it's 16
                                                        // elements. This becomes the vector width of
                                                        // math instructions in the epilogue too
        ElementAccumulator,                                // <- data type of accumulator
        ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function
        // Number of pipelines you want to use
        constexpr int NumStages = 4;

        using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                                LayoutInputA,
                                                ElementInputB,
                                                LayoutInputB,
                                                ElementOutput,
                                                LayoutOutput,
                                                ElementAccumulator,
                                                MMAOp,
                                                SmArch,
                                                ShapeMMAThreadBlock,
                                                ShapeMMAWarp,
                                                ShapeMMAOp,
                                                EpilogueOp,
                                                SwizzleThreadBlock,
                                                NumStages>;


    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(Ga.sizes()[0], Gb0.sizes()[0], Ga.sizes()[1]);


    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {(float*)Ga.data_ptr(), problem_size.k()},  // <- reference to matrix A on device
                                        {(float*)Gb0.data_ptr(), problem_size.k()}, // <- reference to matrix B on device
                                        {},  // <- reference to matrix C on device
                                        {(float*)Gd0.data_ptr(), problem_size.n()},  // <- reference to matrix D on device
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op();
    CUTLASS_CHECK(status);
}
void gemm_fp32_relu(torch::Tensor Ga, torch::Tensor Gb0,  torch::Tensor Gd0){
     // This code section describes the epilogue part of the kernel
       using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
        // Number of pipelines you want to use
        constexpr int NumStages = 4;

        using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                                LayoutInputA,
                                                ElementInputB,
                                                LayoutInputB,
                                                ElementOutput,
                                                LayoutOutput,
                                                ElementAccumulator,
                                                MMAOp,
                                                SmArch,
                                                ShapeMMAThreadBlock,
                                                ShapeMMAWarp,
                                                ShapeMMAOp,
                                                EpilogueOp,
                                                SwizzleThreadBlock,
                                                NumStages>;


    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(Ga.sizes()[0], Gb0.sizes()[0], Ga.sizes()[1]);


    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {(float*)Ga.data_ptr(), problem_size.k()},  // <- reference to matrix A on device
                                        {(float*)Gb0.data_ptr(), problem_size.k()}, // <- reference to matrix B on device
                                        {},  // <- reference to matrix C on device
                                        {(float*)Gd0.data_ptr(), problem_size.n()},  // <- reference to matrix D on device
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op();
    CUTLASS_CHECK(status);
}
void gemm_fp32(torch::Tensor Ga, torch::Tensor Gb0,  torch::Tensor Gd0,  bool relu) {


    if (relu){
       gemm_fp32_relu(Ga, Gb0, Gd0);
    }
    else{
        gemm_fp32_norelu(Ga, Gb0, Gd0);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &gemm_fp32, "gemm based on cutlass", py::arg("input"), py::arg("weight")
    , py::arg("output"), py::arg("relu"));
    //"Gpu_Cublas"代表python中对应的函数，&np_multiply_Cublas是对应的C++函数指针，之后的字符串是python中的函数doc
}
