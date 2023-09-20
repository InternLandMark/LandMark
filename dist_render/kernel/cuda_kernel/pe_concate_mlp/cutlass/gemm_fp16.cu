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

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"
#include "helper.h"

using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;

ElementCompute alpha0 = ElementCompute(1);
ElementCompute beta0 = ElementCompute(0); //beta=1 for bias
// attention: 这里的第二项"128"需要与problem_size_0.n()对齐, 但不可使用变量, 必须手动设置
using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
// attention: 这里的第二项"128"需要与gemm_f16_sm80_problem_size_1.n()对齐, 但不可使用变量, 必须手动设置
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

////////////////////////////////////////////////////////////////////////////////
void gemm_fp16_norelu(torch::Tensor Ga, torch::Tensor Gb0,  torch::Tensor Gd0){
  cutlass::gemm::GemmCoord problem_size_0(Ga.sizes()[0], Gb0.sizes()[0], Ga.sizes()[1]);
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
      ElementCompute>;
  using Gemm0 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor, // A
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // B
    ElementOutput,
    cutlass::layout::RowMajor,    // C
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;


    cutlass::half_t* d_at = (cutlass::half_t*)Ga.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)Gb0.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*)Gd0.data_ptr();


    typename Gemm0::Arguments arguments_0(
        problem_size_0,
        {d_at, problem_size_0.k()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size_0.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size_0.n()},
        {alpha0, beta0}
      );
      Gemm0 gemm_op_0;
      // cudaEvent_t start, stop1, stop2;
      //     cudaEventCreate(&start);
      //     cudaEventCreate(&stop1);

      //     cudaEventRecord(start);

      cutlass::Status status = gemm_op_0.initialize(arguments_0);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op_0();
      CUTLASS_CHECK(status);
}


void gemm_fp16_relu(torch::Tensor Ga, torch::Tensor Gb0,  torch::Tensor Gd0){
  cutlass::gemm::GemmCoord problem_size_0(Ga.sizes()[0], Gb0.sizes()[0], Ga.sizes()[1]);
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using Gemm0 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor, // A
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // B
    ElementOutput,
    cutlass::layout::RowMajor,    // C
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;


    cutlass::half_t* d_at = (cutlass::half_t*)Ga.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)Gb0.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*)Gd0.data_ptr();


    typename Gemm0::Arguments arguments_0(
        problem_size_0,
        {d_at, problem_size_0.k()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size_0.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size_0.n()},
        {alpha0, beta0}
      );
      Gemm0 gemm_op_0;
      // cudaEvent_t start, stop1, stop2;
      //     cudaEventCreate(&start);
      //     cudaEventCreate(&stop1);

      //     cudaEventRecord(start);

      cutlass::Status status = gemm_op_0.initialize(arguments_0);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op_0();
      CUTLASS_CHECK(status);
}

void gemm_fp16(torch::Tensor Ga, torch::Tensor Gb0,  torch::Tensor Gd0, bool relu) {
    if (relu){
      gemm_fp16_relu(Ga, Gb0, Gd0);
    }
    else{
      gemm_fp16_norelu(Ga, Gb0, Gd0);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &gemm_fp16, "gemm based on cutlass", py::arg("input"), py::arg("weight")
    , py::arg("output"), py::arg("relu"));
    //"Gpu_Cublas"代表python中对应的函数，&np_multiply_Cublas是对应的C++函数指针，之后的字符串是python中的函数doc
}



////////////////////////////////////////////////////////////////////////////////
