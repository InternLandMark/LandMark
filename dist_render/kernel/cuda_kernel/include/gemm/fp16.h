#pragma once

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
// attention: 这里的第二项"128"需要与problem_size.n()对齐, 但不可使用变量, 必须手动设置
using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
// attention: 这里的第二项"128"需要与gemm_f16_sm80_problem_size_1.n()对齐, 但不可使用变量, 必须手动设置
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;



////////////////////////////////////////////////////////////////////////////////
void gemm_fp16_norelu(torch::Tensor input, torch::Tensor weight,  torch::Tensor Gd0){
  cutlass::gemm::GemmCoord problem_size(input.sizes()[0], weight.sizes()[1], input.sizes()[1]);
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
      ElementCompute>;
   using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // A
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
    cutlass::half_t* d_at = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*)Gd0.data_ptr();
    typename Gemm::Arguments arguments(
        problem_size,
        {d_at, problem_size.m()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size.n()},
        {alpha0, beta0}
      );
      Gemm gemm_op;
      cutlass::Status status = gemm_op.initialize(arguments);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op();
      CUTLASS_CHECK(status);
}


/*
input : column major
weight : column major
output : row major
k shoule be a multiple of 4
n should be a multiple of 8
can personalize the stream
*/

void gemm_fp16_col_col_row_relu_4x_multiStreams(torch::Tensor input, torch::Tensor weight,  half* output, cudaStream_t stream){

  cutlass::gemm::GemmCoord problem_size(input.size(0), weight.size(0), input.size(1));
  // using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<  //error: no operator "-" matches these operands
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // A
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
    2
  >;



    cutlass::half_t* d_at = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*) output;


    typename Gemm::Arguments arguments(
        problem_size,
        {d_at, problem_size.m()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size.n()},
        {alpha0, beta0}
      );
      Gemm gemm_op;

      // // Using the arguments, query for extra workspace required for matrix multiplication computation
      // size_t workspace_size = Gemm::get_workspace_size(arguments);

      // // Allocate workspace memory
      //  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
      // cutlass::Status status = gemm_op.initialize(arguments, workspace.get(), stream);

      cutlass::Status status = gemm_op.initialize(arguments);
      CUTLASS_CHECK(status);
      // Run the GEMM 是否需要在这里增加stream？
      status = gemm_op(stream);
      CUTLASS_CHECK(status);
}

void gemm_fp16_row_col_row_sigmoid_4x_problemsize(torch::Tensor input, torch::Tensor weight,  half* output,
 const int m, const int n, const int k){

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSigmoid<  //error: no operator "-" matches these operands
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using Gemm = cutlass::gemm::device::Gemm<
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
    4
  >;



    cutlass::half_t* d_at = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*) output;


    typename Gemm::Arguments arguments(
        problem_size,
        {d_at, problem_size.k()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size.n()},
        {alpha0, beta0}
      );
      Gemm gemm_op;


      cutlass::Status status = gemm_op.initialize(arguments);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op();
      CUTLASS_CHECK(status);
}


void gemm_fp16_row_col_row_relu_4x_problemsize(torch::Tensor input, torch::Tensor weight,  half* output,
 const int m, const int n, const int k){

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<  //error: no operator "-" matches these operands
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using Gemm = cutlass::gemm::device::Gemm<
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
    4
  >;



    cutlass::half_t* d_at = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*) output;


    typename Gemm::Arguments arguments(
        problem_size,
        {d_at, problem_size.k()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size.n()},
        {alpha0, beta0}
      );
      Gemm gemm_op;


      cutlass::Status status = gemm_op.initialize(arguments);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op();
      CUTLASS_CHECK(status);
}


void gemm_fp16_col_col_row_relu_4x_problemsize(torch::Tensor input, torch::Tensor weight,  half* output,
    const int m, const int n, const int k){

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<  //error: no operator "-" matches these operands
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // A
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
    4
  >;



    cutlass::half_t* d_at = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*) output;


    typename Gemm::Arguments arguments(
        problem_size,
        {d_at, problem_size.m()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size.n()},
        {alpha0, beta0}
      );
      Gemm gemm_op;


      cutlass::Status status = gemm_op.initialize(arguments);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op();
      CUTLASS_CHECK(status);
}

void gemm_fp16_col_col_col_relu_4x_problemsize(torch::Tensor input, torch::Tensor weight,  half* output,
    const int m, const int n, const int k){

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<  //error: no operator "-" matches these operands
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // A
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // B
    ElementOutput,
    cutlass::layout::ColumnMajor,    // C
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    4
  >;



    cutlass::half_t* d_at = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*) output;


    typename Gemm::Arguments arguments(
        problem_size,
        {d_at, problem_size.m()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size.m()},
        {alpha0, beta0}
      );
      Gemm gemm_op;


      cutlass::Status status = gemm_op.initialize(arguments);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op();
      CUTLASS_CHECK(status);
}

void gemm_fp16_col_col_row_sigmoid_4x_problemsize(torch::Tensor input, torch::Tensor weight,  half* output,
    const int m, const int n, const int k){

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSigmoid<  //error: no operator "-" matches these operands
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // A
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
    4
  >;



    cutlass::half_t* d_at = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*) output;


    typename Gemm::Arguments arguments(
        problem_size,
        {d_at, problem_size.m()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size.n()},
        {alpha0, beta0}
      );
      Gemm gemm_op;


      cutlass::Status status = gemm_op.initialize(arguments);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op();
      CUTLASS_CHECK(status);
}

void gemm_fp16_col_col_row_relu_4x(torch::Tensor input, torch::Tensor weight,  half* output){

  cutlass::gemm::GemmCoord problem_size(input.size(0), weight.size(0), input.size(1));
  // using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<  //error: no operator "-" matches these operands
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // A
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
    4
  >;



    cutlass::half_t* d_at = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*) output;


    typename Gemm::Arguments arguments(
        problem_size,
        {d_at, problem_size.m()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size.n()},
        {alpha0, beta0}
      );
      Gemm gemm_op;


      cutlass::Status status = gemm_op.initialize(arguments);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op();
      CUTLASS_CHECK(status);
}

void gemm_fp16_sigmoid(torch::Tensor input, torch::Tensor weight,  torch::Tensor Gd0){
  cutlass::gemm::GemmCoord problem_size(input.sizes()[0], weight.sizes()[1], input.sizes()[1]);
  // using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSigmoid<  //error: no operator "-" matches these operands
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
   using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::ColumnMajor, // A
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



    cutlass::half_t* d_at = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* d_b0t = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* d_d0t = (cutlass::half_t*)Gd0.data_ptr();


    typename Gemm::Arguments arguments(
        problem_size,
        {d_at, problem_size.m()},
        // tensor_B0.device_ref(),
        {d_b0t, problem_size.k()},
        {},
        // tensor_D0.device_ref(),
        {d_d0t, problem_size.n()},
        {alpha0, beta0}
      );
      Gemm gemm_op;


      cutlass::Status status = gemm_op.initialize(arguments);
      CUTLASS_CHECK(status);
      // Run the GEMM
      status = gemm_op();
      CUTLASS_CHECK(status);
}

std::function<void(at::Tensor, at::Tensor, half *, int, int, int)> select_gemm( bool in_rowmajor, bool out_rowmajor, std::string activation){
    if (in_rowmajor && out_rowmajor && activation == "Relu"){
      return std::function<void(at::Tensor, at::Tensor, half *, int, int, int)>(gemm_fp16_row_col_row_relu_4x_problemsize);
    }
    if (in_rowmajor && out_rowmajor && activation == "Sigmoid"){
      return std::function<void(at::Tensor, at::Tensor, half *, int, int, int)>(gemm_fp16_row_col_row_sigmoid_4x_problemsize);
    }
    if (!in_rowmajor && out_rowmajor && activation == "Relu"){
      return std::function<void(at::Tensor, at::Tensor, half *, int, int, int)>(gemm_fp16_col_col_row_relu_4x_problemsize);
    }
    if (!in_rowmajor && !out_rowmajor && activation == "Relu"){
      return std::function<void(at::Tensor, at::Tensor, half *, int, int, int)>(gemm_fp16_col_col_col_relu_4x_problemsize);
    }
    if (!in_rowmajor && out_rowmajor && activation == "Sigmoid"){
      return std::function<void(at::Tensor, at::Tensor, half *, int, int, int)>(gemm_fp16_col_col_row_sigmoid_4x_problemsize);
    }
    std::cerr << "Error: GEMM Condition not met!\n";
    std::abort();
}
