/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_relu.h"

#include "reference/device/tensor_scale_bias.h"
#include "helper.h"

#define CHECK_GT(val1, val2) \
    if((val1) <= (val2)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_GT failed\n";
#define CHECK_TRUE(val) \
    if(!(val)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_TRUE failed\n";



template <typename B2bGemm_, int InterleavedK_>
struct B2bInterleavedFusedGemmRun
{

  using B2bGemm = B2bGemm_;
  using ElementAccumulator = typename B2bGemm::ElementAccumulator;
  using ElementCompute = typename B2bGemm::B2bGemmKernel::Epilogue::OutputOp::ElementCompute;



  //
  // Methods
  //


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

    //
    // Initialize the GEMM operator
    //

    typename B2bGemm::Arguments arguments{
      problem_size_0,
      problem_size_1,
      tensor_A0.device_ref(),
      tensor_B0_reordered.device_ref(),
      tensor_C0.device_ref(),
      tensor_Scale0.device_ref(),
      tensor_Bias0.device_ref(),
      tensor_B1_reordered.device_ref(),
      {tensor_Bias1.device_data(), typename B2bGemm::LayoutC::Stride(0)},
      tensor_D1.device_ref(),
      {alpha0, beta0},
      {alpha1, beta1},
    };

    B2bGemm b2b_gemm_op;

    cutlass::Status status = b2b_gemm_op.can_implement(arguments);

    if(status != cutlass::Status::kSuccess) {
        std::cout << "Problem sizes not supported.\n"
                << "Requirments:\n"
                << "    problem_size_0.M = problem_size_1.M\n"
                << "    problem_size_0.N = problem_size_1.K\n"
                << "    ThreadblockShape0::kN = problem_size_0.N\n"
                << "    ThreadblockShape1::kN = problem_size_1.N" << std::endl;
    }

    status = b2b_gemm_op.initialize(arguments);

    CUTLASS_CHECK(status);

    for(int i = 0; i < warm_ups; i++) {
        status = b2b_gemm_op();
        CUTLASS_CHECK(status);
    }

    //
    // Run the GEMM
    //

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i < runs; i++) {
        status = b2b_gemm_op();

        CUTLASS_CHECK(status);
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float gemmTime;
    cudaEventElapsedTime(&gemmTime, start, stop);
    std::cout << "Fusion time " << gemmTime / (float)runs << " ms\n";

    tensor_D1.sync_host();

    //
    // Verify
    //

    cutlass::reference::device::Gemm<
        typename B2bGemm::ElementA, typename B2bGemm::LayoutA,
        typename B2bGemm::ElementB, typename B2bGemm::LayoutB,
        ElementAccumulator, typename B2bGemm::LayoutC,
        ElementAccumulator, ElementAccumulator>
        reference_gemm_0;

    cutlass::reference::device::Gemm<
        typename B2bGemm::ElementA, typename B2bGemm::LayoutA,
        typename B2bGemm::ElementB, typename B2bGemm::LayoutB,
        typename B2bGemm::ElementC, typename B2bGemm::LayoutC, ElementCompute,
        ElementAccumulator, typename B2bGemm::Operator>
        reference_gemm_1;

    reference_gemm_0(
      problem_size_0,
      ElementAccumulator(1), //intermediate alpha=1
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      ElementAccumulator(0), //beta = 0
      reference_Z0.device_ref(),
      reference_Z0.device_ref(),
      ElementAccumulator(0)
    );

    cutlass::reference::device::TensorScaleBiasGemm<
      ElementAccumulator, typename B2bGemm::ElementC, typename B2bGemm::LayoutC,
      ElementCompute, typename B2bGemm::LayoutScaleBias
    > (
      problem_size_0,
      reference_Z0.device_ref(),
      reference_D0.device_ref(),
      alpha0,
      tensor_Scale0.device_ref(),
      tensor_Bias0.device_ref()
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D0.device_view());
    }

    reference_gemm_1(
      problem_size_1,
      alpha1,
      reference_D0.device_ref(),
      tensor_B1.device_ref(),
      beta1,
      {tensor_Bias1.device_data(), typename B2bGemm::LayoutC::Stride(0)},
      reference_D1.device_ref()
    );
    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D1.device_view());
    }
    cudaDeviceSynchronize();
    reference_D0.sync_host();
    reference_D1.sync_host();

    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D0.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D1.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(
      reference_D1.host_view(),
      tensor_D1.host_view());

    CHECK_TRUE(passed);
    if (!passed)
    {

      std::stringstream fname;

      fname << "error_B2bGemm_device_interleaved_fused.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream file(fname.str());

      file
        << "A0 =\n" << tensor_A0.host_view()
        << "\nB0 =\n" << tensor_B0.host_view()
        << "\nB0_reordered =\n" << tensor_B0_reordered.host_view()
        << "\nC0 =\n" << tensor_C0.host_view()
        << "\nScale0:\n" << tensor_Scale0.host_view() << "\n"
        << "\nBias0:\n" << tensor_Bias0.host_view() << "\n"
        << "\nB1 =\n" << tensor_B1.host_view()
        << "\nB1_reordered =\n" << tensor_B1_reordered.host_view()
        << "\nC1 =\n" << tensor_C1.host_view()
        << "\nBias1:\n" << tensor_Bias1.host_view() << "\n"
        << "\n\nReference =\n" << reference_D1.host_view()
        << "\nComputed =\n" << tensor_D1.host_view();
    }
    return passed;
  }

};

////////////////////////////////////////////////////////////////////////////////
