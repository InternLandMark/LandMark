#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

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
    cutlass::gemm::GemmShape<128, 64, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

///////////////////////////////////////////////////////////////////////////////////////////////////
template < typename Gemm_3xTF32_>
struct MlpRenderGemmRun
{

    using Gemm_3xTF32 = Gemm_3xTF32_;

    bool run(cutlass::gemm::GemmCoord problem_size,
            float* input, float* weight, float* output,
            float alpha, float beta, int split_k_slices){
         typename Gemm_3xTF32::Arguments arguments_3xtf32{problem_size,  // <- problem size of matrix multiplication
                                     {input, problem_size.k()},  // <- reference to matrix A on device
                                      {weight, problem_size.k()}, // <- reference to matrix B on device
                                      {},  // <- reference to matrix C on device
                                      {output, problem_size.n()},  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor
        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size_3xtf32 = Gemm_3xTF32::get_workspace_size(arguments_3xtf32);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace_3xtf32(workspace_size_3xtf32);

        // Instantiate CUTLASS kernel depending on templates
        Gemm_3xTF32 gemm_op_3xTF32;

        // Check the problem size is supported or not
        cutlass::Status status_3xtf32 = gemm_op_3xTF32.can_implement(arguments_3xtf32);
        CUTLASS_CHECK(status_3xtf32);

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status_3xtf32 = gemm_op_3xTF32.initialize(arguments_3xtf32, workspace_3xtf32.get());
        CUTLASS_CHECK(status_3xtf32);


        status_3xtf32 = gemm_op_3xTF32();
        CUTLASS_CHECK(status_3xtf32);

        return true;
    }
};
