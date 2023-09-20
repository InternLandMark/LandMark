/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once
#include <torch/extension.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_relu.h"
#include "reference/device/tensor_scale_bias.h"
#include "helper.h"



inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

template <typename Gemm0_, typename Gemm1_>
struct B2bNonFusedGemmRun
{

  using Gemm0 = Gemm0_;
  using Gemm1 = Gemm1_;
  using ElementAccumulator = typename Gemm0::ElementAccumulator;
  using ElementCompute = typename Gemm0::GemmKernel::Epilogue::OutputOp::ElementCompute;



  /// Executes one test
  bool run(
    cutlass::gemm::GemmCoord problem_size_0,
    cutlass::gemm::GemmCoord problem_size_1,
    cutlass::half_t* d_at,
    cutlass::half_t* d_b0t,
    cutlass::half_t* d_b1t,
    cutlass::half_t* d_d0t,
    cutlass::half_t* d_d1t,
    cutlass::half_t* d_drt,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(0),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(0),
    bool relu = true) {
    // // cout << d_At.


    //
    // Initialize the GEMM operator
    //
    // cutlass::half_t *A;               //申明A矩阵host端指针
    // size_t A_mem_size = sizeof(cutlass::half_t) * M * K; //memory size of matrix A = M * K * sizeof(float)
    // A = (cutlass::half_t*)malloc(A_mem_size);  // host端A矩阵分配内存
    // generate_tensor_2D(A, M, K);     // 填充A矩阵
    // cutlass::half_t *d_A;            // 申明device端A矩阵的指针
    // cudaMalloc((void**)&d_A, A_mem_size);  // device端为A矩阵分配内存
    // cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice); // 将矩阵A的数据传递到device端


    // std::cout << d_At << endl;
    // std::cout << d_A << endl;
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

    typename Gemm1::Arguments arguments_1{
      problem_size_1,
      // tensor_D0.device_ref(),
      {d_d0t, problem_size_1.k()},
      // tensor_B1.device_ref(),
      {d_b1t, problem_size_1.k()},
      {},
      // tensor_D1.device_ref(),
      {d_d1t, problem_size_1.n()},
      {alpha1, beta1}
    };


    Gemm0 gemm_op_0;
    Gemm1 gemm_op_1;
    // todo: 之后需要将其复用, 不可放在此
    cutlass::Status status = gemm_op_0.initialize(arguments_0);
    CUTLASS_CHECK(status);
    status = gemm_op_1.initialize(arguments_1);
    CUTLASS_CHECK(status);
    // Run the GEMM
    status = gemm_op_0();
    CUTLASS_CHECK(status);
    status = gemm_op_1();
    CUTLASS_CHECK(status);

  }
};
