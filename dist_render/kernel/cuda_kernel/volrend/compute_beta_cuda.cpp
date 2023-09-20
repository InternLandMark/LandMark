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

// CUDA funciton declearition
std::vector<torch::Tensor> compute_beta_cuda(
    torch::Tensor sigma, torch::Tensor z_val, torch::Tensor beta, float distance_scale);
    // torch::Tensor sigma, torch::Tensor dist, torch::Tensor beta);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> compute_beta(
    torch::Tensor sigma, torch::Tensor z_val, torch::Tensor beta, float distance_scale)
    // torch::Tensor sigma, torch::Tensor dist, torch::Tensor beta)
{
    // CHECK_INPUT(sigma);
    // CHECK_INPUT(dist);
    // CHECK_INPUT(z_val);
    // CHECK_INPUT(beta);

    // return compute_beta_cuda(sigma, dist, beta);
    return compute_beta_cuda(sigma, z_val, beta, distance_scale);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda", &compute_beta, "compute_beta (CUDA)");
}
