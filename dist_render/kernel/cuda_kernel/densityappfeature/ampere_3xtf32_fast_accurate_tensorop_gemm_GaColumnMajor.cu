

/**
NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.

We can use the tf32 mode of tensor core to emulate a fast accurate SGEMM kernel which is accelerated
using Ampere Tensor Cores (see include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h).

The trick is very simple
  a x b = (a_big + a_small) x (b_big + b_small) = a_big x b_big + a_big x b_small + a_small x b_big
  big = convert_to_tf32(fp32)
  small = convert_to_tf32(fp32 - big)

a_small x b_small is discarded because they are too small.

This example demonstrates usage of this kernel, along with accuracy measurements w.r.t. actual FP32
results (SGEMM using SIMT) and against FP64 results (DGEMM)

To enable this feature, the only change needs to make is to change the default OpMultiplyAdd to
OpMultiplyAddFastF32.

Now, we have several different flavors of sgemm now in the profiler for Ampere.  Here are the difference

  sgemm           // CUDA core SIMT kernel.  FP32 in, accumulated in FP32, FP32 out.
  s1688gemm       // Use 3xTF32 to emulate FP32.  FP32 in, converted in TF32-big and TF32-small internally,
                  // accumulated in FP32, FP32 out.
  s1688tf32gemm   // Use 1xTF32.  FP32 in, converted to one TF32 internally, accumulated in FP32, FP32 out.
  s1688gemm_tf32  // TF32 in, accumulated in FP32, FP32 out.
*/
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
#include <vector>
#include <limits>

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
#include "ampere_3xtf32_fast_accurate_tensorop_gemm_GaColumnMajor.h"



bool gemm(torch::Tensor Ga, torch::Tensor Gb0,  torch::Tensor Gd0, bool relu) {
  // Create a tuple of problem size for matrix multiplication
  using LayoutInputA = cutlass::layout::ColumnMajor;
  int m = Ga.sizes()[1]; int n = Gb0.sizes()[0]; int k = Ga.sizes()[0];
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  // Number of pipelines you want to use
  const int NumStages = 3;
  float alpha = 1; float beta = 0; int split_k_slices = 1;
  if (m % 4 == 0 & n % 4 == 0 & k % 4 == 0) {
     if (relu){
        using Epilogue = cutlass::epilogue::thread::LinearCombinationRelu<
                float,                                     // <- data type of output matrix
                4,
                                                                  // memory access. For a byte, it's 16
                                                                  // elements. This becomes the vector width of
                                                                  // math instructions in the epilogue too
                float,                                   // <- data type of accumulator
                float>;

        using Gemm_3xTF32 = cutlass::gemm::device::Gemm<
                                            float,
                                            LayoutInputA,
                                            float,
                                            LayoutInputB,
                                            float,
                                            LayoutOutput,
                                            float,
                                            MMAOp,
                                            SmArch,
                                            ShapeMMAThreadBlock,
                                            ShapeMMAWarp,
                                            ShapeMMAOp,
                                            Epilogue,
                                            SwizzleThreadBlock,
                                            NumStages,
                                            4,
                                            4,
                                            false,
                                            cutlass::arch::OpMultiplyAddFastF32>;
        MlpRenderGemmRun<Gemm_3xTF32> Gemm;
        Gemm.run(problem_size, (float*)Ga.data_ptr(), (float*)Gb0.data_ptr(), (float*)Gd0.data_ptr(), alpha, beta,split_k_slices);
      }
      else{
         using Epilogue = cutlass::epilogue::thread::LinearCombination<
                float,                                     // <- data type of output matrix
                4,
                                                                  // memory access. For a byte, it's 16
                                                                  // elements. This becomes the vector width of
                                                                  // math instructions in the epilogue too
                float,                                   // <- data type of accumulator
                float>;
         using Gemm_3xTF32 = cutlass::gemm::device::Gemm<
                                            float,
                                            LayoutInputA,
                                            float,
                                            LayoutInputB,
                                            float,
                                            LayoutOutput,
                                            float,
                                            MMAOp,
                                            SmArch,
                                            ShapeMMAThreadBlock,
                                            ShapeMMAWarp,
                                            ShapeMMAOp,
                                            Epilogue,
                                            SwizzleThreadBlock,
                                            NumStages,
                                            4,
                                            4,
                                            false,
                                            cutlass::arch::OpMultiplyAddFastF32>;
          MlpRenderGemmRun<Gemm_3xTF32> Gemm;
  Gemm.run(problem_size, (float*)Ga.data_ptr(), (float*)Gb0.data_ptr(), (float*)Gd0.data_ptr(), alpha, beta,split_k_slices);
      }
  }
  else if (m % 2 == 0 & n % 2 == 0 & k % 2 == 0) {
     if (relu){
        using Epilogue = cutlass::epilogue::thread::LinearCombinationRelu<
                float,                                     // <- data type of output matrix
                2,
                                                                  // memory access. For a byte, it's 16
                                                                  // elements. This becomes the vector width of
                                                                  // math instructions in the epilogue too
                float,                                   // <- data type of accumulator
                float>;

        using Gemm_3xTF32 = cutlass::gemm::device::Gemm<
                                            float,
                                            LayoutInputA,
                                            float,
                                            LayoutInputB,
                                            float,
                                            LayoutOutput,
                                            float,
                                            MMAOp,
                                            SmArch,
                                            ShapeMMAThreadBlock,
                                            ShapeMMAWarp,
                                            ShapeMMAOp,
                                            Epilogue,
                                            SwizzleThreadBlock,
                                            NumStages,
                                            2,
                                            2,
                                            false,
                                            cutlass::arch::OpMultiplyAddFastF32>;
        MlpRenderGemmRun<Gemm_3xTF32> Gemm;
        Gemm.run(problem_size, (float*)Ga.data_ptr(), (float*)Gb0.data_ptr(), (float*)Gd0.data_ptr(), alpha, beta,split_k_slices);
      }
      else{
         using Epilogue = cutlass::epilogue::thread::LinearCombination<
                float,                                     // <- data type of output matrix
                2,
                                                                  // memory access. For a byte, it's 16
                                                                  // elements. This becomes the vector width of
                                                                  // math instructions in the epilogue too
                float,                                   // <- data type of accumulator
                float>;
         using Gemm_3xTF32 = cutlass::gemm::device::Gemm<
                                            float,
                                            LayoutInputA,
                                            float,
                                            LayoutInputB,
                                            float,
                                            LayoutOutput,
                                            float,
                                            MMAOp,
                                            SmArch,
                                            ShapeMMAThreadBlock,
                                            ShapeMMAWarp,
                                            ShapeMMAOp,
                                            Epilogue,
                                            SwizzleThreadBlock,
                                            NumStages,
                                            2,
                                            2,
                                            false,
                                            cutlass::arch::OpMultiplyAddFastF32>;
          MlpRenderGemmRun<Gemm_3xTF32> Gemm;
  Gemm.run(problem_size, (float*)Ga.data_ptr(), (float*)Gb0.data_ptr(), (float*)Gd0.data_ptr(), alpha, beta,split_k_slices);
      }
  }
  else{
      if (relu){
        using Epilogue = cutlass::epilogue::thread::LinearCombinationRelu<
                float,                                   // <- data type of output matrix
                1,
                                                                  // memory access. For a byte, it's 16
                                                                  // elements. This becomes the vector width of
                                                                  // math instructions in the epilogue too
                float,                                   // <- data type of accumulator
                float>;

        using Gemm_3xTF32 = cutlass::gemm::device::Gemm<
                                            float,
                                            LayoutInputA,
                                            float,
                                            LayoutInputB,
                                            float,
                                            LayoutOutput,
                                            float,
                                            MMAOp,
                                            SmArch,
                                            ShapeMMAThreadBlock,
                                            ShapeMMAWarp,
                                            ShapeMMAOp,
                                            Epilogue,
                                            SwizzleThreadBlock,
                                            NumStages,
                                            1,
                                            1,
                                            false,
                                            cutlass::arch::OpMultiplyAddFastF32>;
        MlpRenderGemmRun<Gemm_3xTF32> Gemm;
        Gemm.run(problem_size, (float*)Ga.data_ptr(), (float*)Gb0.data_ptr(), (float*)Gd0.data_ptr(), alpha, beta,split_k_slices);
      }
      else{
         using Epilogue = cutlass::epilogue::thread::LinearCombination<
                float,                                     // <- data type of output matrix
                1,
                                                                  // memory access. For a byte, it's 16
                                                                  // elements. This becomes the vector width of
                                                                  // math instructions in the epilogue too
                float,                                   // <- data type of accumulator
                float>;
         using Gemm_3xTF32 = cutlass::gemm::device::Gemm<
                                            float,
                                            LayoutInputA,
                                            float,
                                            LayoutInputB,
                                            float,
                                            LayoutOutput,
                                            float,
                                            MMAOp,
                                            SmArch,
                                            ShapeMMAThreadBlock,
                                            ShapeMMAWarp,
                                            ShapeMMAOp,
                                            Epilogue,
                                            SwizzleThreadBlock,
                                            NumStages,
                                            1,
                                            1,
                                            false,
                                            cutlass::arch::OpMultiplyAddFastF32>;
          MlpRenderGemmRun<Gemm_3xTF32> Gemm;
         Gemm.run(problem_size, (float*)Ga.data_ptr(), (float*)Gb0.data_ptr(), (float*)Gd0.data_ptr(), alpha, beta,split_k_slices);
    }
  }
    // if (n % 4 == 0 & k % 1 == 0) {
    //   build_gemm( Ga,  Gb0,   Gd0, 4, 1, relu, alpha, beta, split_k_slices, NumStages );
    // }
    // if (n % 2 == 0 & k % 4 == 0) {
    //   build_gemm( Ga,  Gb0,   Gd0, 2, 4, relu, alpha, beta, split_k_slices, NumStages );
    // }
    // if (n % 2 == 0 & k % 2 == 0) {
    //   build_gemm( Ga,  Gb0,   Gd0, 2, 2, relu, alpha, beta, split_k_slices, NumStages );
    // }
    // if (n % 2 == 0 & k % 1 == 0) {
    //   build_gemm( Ga,  Gb0,   Gd0, 2, 1, relu, alpha, beta, split_k_slices, NumStages );
    // }
    // if (n % 1 == 0 & k % 4 == 0) {
    //   build_gemm( Ga,  Gb0,   Gd0, 1, 4, relu, alpha, beta, split_k_slices, NumStages );
    // }
    // if (n % 1 == 0 & k % 2 == 0) {
    //   build_gemm( Ga,  Gb0,   Gd0, 1, 2, relu, alpha, beta, split_k_slices, NumStages );
    // }
    // if (n % 1 == 0 & k % 1 == 0) {
    //   build_gemm( Ga,  Gb0,   Gd0, 1, 1, relu, alpha, beta, split_k_slices, NumStages );
    // }

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &gemm, "gemm based on  cutlass", py::arg("input"), py::arg("weight")
    , py::arg("output"), py::arg("relu"));
}
