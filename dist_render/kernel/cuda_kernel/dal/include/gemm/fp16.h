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
# define debug 0

using ElementInput= cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;

ElementCompute alpha0 = ElementCompute(1);
ElementCompute beta0 = ElementCompute(0);
using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;


using EpilogueOpSigmoid = cutlass::epilogue::thread::LinearCombinationSigmoid<  //error: no operator "-" matches these operands
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

using EpilogueOpRelu = cutlass::epilogue::thread::LinearCombinationRelu<  //error: no operator "-" matches these operands
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementCompute>;  // <- data type for alpha/beta in linear combination function


using EpilogueOpSigmoid_1align = cutlass::epilogue::thread::LinearCombinationSigmoid<  //error: no operator "-" matches these operands
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

using EpilogueOpRelu_1align = cutlass::epilogue::thread::LinearCombinationRelu<  //error: no operator "-" matches these operands
    ElementOutput,
    1,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

using EpilogueOp_1align = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    1,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementCompute>;  // <- data type for alpha/beta in linear combination function



template <typename EpilogueOp, typename LayoutInput, typename LayoutWeight, typename LayoutOutput>
void gemm_fp16_row_col_row_4x_problemsize(cutlass::half_t* input_ptr, cutlass::half_t* weight_ptr, cutlass::half_t* output_ptr,
    int m, int n, int k, const std::optional<cudaStream_t> stream = std::nullopt){

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    using Gemm = cutlass::gemm::device::Gemm<
      ElementInput,
      LayoutInput, // A
      cutlass::half_t,
      LayoutWeight, // B
      ElementOutput,
      LayoutOutput,    // C
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

    typename Gemm::Arguments arguments(
        problem_size,
        {input_ptr, problem_size.k()},
        {weight_ptr, problem_size.k()},
        {},
        {output_ptr, problem_size.n()},
        {alpha0, beta0}
    );
    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(arguments);
    CUTLASS_CHECK(status);
    if (stream)  gemm_op(stream.value());
    else status = gemm_op();
    CUTLASS_CHECK(status);
}

template <typename EpilogueOp, typename LayoutInput, typename LayoutWeight, typename LayoutOutput>
void gemm_fp16_col_col_row_4x_problemsize(cutlass::half_t* input_ptr, cutlass::half_t* weight_ptr, cutlass::half_t* output_ptr,
    int m, int n, int k, const std::optional<cudaStream_t> stream = std::nullopt){

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    using Gemm = cutlass::gemm::device::Gemm<
      ElementInput,
      LayoutInput, // A
      cutlass::half_t,
      LayoutWeight, // B
      ElementOutput,
      LayoutOutput,    // C
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

    typename Gemm::Arguments arguments(
        problem_size,
        {input_ptr, problem_size.m()},
        {weight_ptr, problem_size.k()},
        {},
        {output_ptr, problem_size.n()},
        {alpha0, beta0}
    );
    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(arguments);
    CUTLASS_CHECK(status);
    if (stream)  gemm_op(stream.value());
    else status = gemm_op();
    CUTLASS_CHECK(status);
}

template <typename EpilogueOp, typename LayoutInput, typename LayoutWeight, typename LayoutOutput>
void gemm_fp16_row_col_col_4x_problemsize(cutlass::half_t* input_ptr, cutlass::half_t* weight_ptr, cutlass::half_t* output_ptr,
    int m, int n, int k, const std::optional<cudaStream_t> stream = std::nullopt){

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    using Gemm = cutlass::gemm::device::Gemm<
      ElementInput,
      LayoutInput, // A
      cutlass::half_t,
      LayoutWeight, // B
      ElementOutput,
      LayoutOutput,    // C
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

    typename Gemm::Arguments arguments(
        problem_size,
        {input_ptr, problem_size.k()},
        {weight_ptr, problem_size.k()},
        {},
        {output_ptr, problem_size.m()},
        {alpha0, beta0}
    );
    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(arguments);
    CUTLASS_CHECK(status);
    if (stream)  gemm_op(stream.value());
    else status = gemm_op();
    CUTLASS_CHECK(status);
}

template <typename EpilogueOp, typename LayoutInput, typename LayoutWeight, typename LayoutOutput>
void gemm_fp16_col_col_col_4x_problemsize(cutlass::half_t* input_ptr, cutlass::half_t* weight_ptr, cutlass::half_t* output_ptr,
    int m, int n, int k, const std::optional<cudaStream_t> stream = std::nullopt){

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    using Gemm = cutlass::gemm::device::Gemm<
      ElementInput,
      LayoutInput, // A
      cutlass::half_t,
      LayoutWeight, // B
      ElementOutput,
      LayoutOutput,    // C
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

    typename Gemm::Arguments arguments(
        problem_size,
        {input_ptr, problem_size.m()},
        {weight_ptr, problem_size.k()},
        {},
        {output_ptr, problem_size.m()},
        {alpha0, beta0}
    );
    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(arguments);
    CUTLASS_CHECK(status);
    if (stream)  gemm_op(stream.value());
    else status = gemm_op();
    CUTLASS_CHECK(status);
}


std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> select_gemm(
  bool in_rowmajor, bool out_rowmajor, std::string activation, int alignment_out = 8){
    if (in_rowmajor && out_rowmajor && activation == "Relu"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>(
      gemm_fp16_row_col_row_4x_problemsize<EpilogueOpRelu, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>);
    }
    else if (!in_rowmajor && out_rowmajor && activation == "Relu"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_col_col_row_4x_problemsize<EpilogueOpRelu, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>);
    }
    else if (in_rowmajor && !out_rowmajor && activation == "Relu"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_row_col_col_4x_problemsize<EpilogueOpRelu, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>);
    }
    else if (!in_rowmajor && !out_rowmajor && activation == "Relu"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_col_col_col_4x_problemsize<EpilogueOpRelu, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>);
    }

    else if (in_rowmajor && out_rowmajor && activation == "Sigmoid"){
      if (alignment_out == 1){
        return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_row_col_row_4x_problemsize<EpilogueOpSigmoid_1align, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>);
      }
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_row_col_row_4x_problemsize<EpilogueOpSigmoid, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>);
    }
    else if (!in_rowmajor && out_rowmajor && activation == "Sigmoid"){
      if (alignment_out == 1){
        return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_col_col_row_4x_problemsize<EpilogueOpSigmoid_1align, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>);
      }
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_col_col_row_4x_problemsize<EpilogueOpSigmoid, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>);
    }
    else if (in_rowmajor && !out_rowmajor && activation == "Sigmoid"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_row_col_col_4x_problemsize<EpilogueOpSigmoid, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>);
    }
    else if (!in_rowmajor && !out_rowmajor && activation == "Sigmoid"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_col_col_col_4x_problemsize<EpilogueOpSigmoid, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>);
    }

    else if (in_rowmajor && out_rowmajor && activation == "None"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_row_col_row_4x_problemsize<EpilogueOp, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>);
    }
    else if (!in_rowmajor && out_rowmajor && activation == "None"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_col_col_row_4x_problemsize<EpilogueOp, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>);
    }
    else if (in_rowmajor && !out_rowmajor && activation == "None"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_row_col_col_4x_problemsize<EpilogueOp, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>);
    }
    else if (!in_rowmajor && !out_rowmajor && activation == "None"){
      return std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)>
      (gemm_fp16_col_col_col_4x_problemsize<EpilogueOp, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>);
    }

    std::cerr << "Error: GEMM Condition not met!\n";
    std::abort();
}
