#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
  CHECK_CUDA(x);                                                                                   \
  CHECK_CONTIGUOUS(x)

#include <unistd.h>
#include <stdlib.h>
#include <chrono>
using namespace std;
using namespace chrono;

#include <time.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

// DO NOT use AT_DISPATCH_XXX_TYPES macro here, they are clumsy
void launch_grid_sample_3d_float(float *input, float *sample_grid, float *output, int n, int d,
                                 int h, int w, int c, int batch_size, bool output_ncdhw, bool isOptExp,
                                 cudaStream_t stream);


void grid_sample(torch::Tensor input, torch::Tensor grid, torch::Tensor output, bool output_ncdhw, bool isOptExpr) {
  // cudaStream_t stream = DeviceMemoryManager::get()->get_move_stream();
  cudaStream_t stream = nullptr;
  CHECK_INPUT(input);
  CHECK_INPUT(grid);
  // CHECK_INPUT(output);
  auto i_shape = input.sizes();
  auto g_shape = grid.sizes();
  auto o_shape = output.sizes();
  if (i_shape.size() == 5) {
    int n = i_shape[0];
    int d = i_shape[1];
    int h = i_shape[2];
    int w = i_shape[3];
    int c = i_shape[4];
    TORCH_CHECK_EQ(o_shape.size(), 5);
    TORCH_CHECK_EQ(o_shape[0], n);
    if (output_ncdhw) {
      TORCH_CHECK_EQ(o_shape[1], c);
    } else {
      TORCH_CHECK_EQ(o_shape[4], c);
    }
    TORCH_CHECK_EQ(g_shape.size(), 5);
    TORCH_CHECK_EQ(g_shape[0], n);
    TORCH_CHECK_EQ(g_shape[4], 3);
    int batch_size = g_shape[1] * g_shape[2] * g_shape[3];
    if (output_ncdhw) {
      TORCH_CHECK_EQ(o_shape[2] * o_shape[3] * o_shape[4], batch_size);
    } else {
      TORCH_CHECK_EQ(o_shape[1] * o_shape[2] * o_shape[3], batch_size);
    }

    switch (input.scalar_type()) {
    case c10::ScalarType::Float:
      launch_grid_sample_3d_float(input.data_ptr<float>(), grid.data_ptr<float>(),
                                  output.data_ptr<float>(), n, d, h, w, c, batch_size, output_ncdhw, isOptExpr,
                                  stream);
      break;
    default:
      throw;
    }
  } else {
  }
  cudaStreamSynchronize(stream);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda", &grid_sample, "compute_grid_sample_and_ewproduct (CUDA)");
//   m.def("cuda2", &compute_grid_sample_and_ewproduct2, "compute_grid_sample_and_ewproduct (CUDA)");
}

