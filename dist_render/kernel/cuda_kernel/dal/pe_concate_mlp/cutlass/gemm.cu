#include <optional>
#include  <stdexcept>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <mma.h>

#include <ATen/AccumulateType.h>

#include <assert.h>
#include <vector>
#include <torch/torch.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"
#include "common/helper.h"

#include "gemm/fp16.h"
#include "gemm/fused3mlp/device/FusedMultiGemmForward.h"


# define blockDIM 256
# define debug 1


void Fused3mlp(
    bool is_in_rowmajor, bool is_layer1out_rowmajor, bool is_layer2out_rowmajor, bool is_output_rowmajor,
    const int M, const int n1, const int k1, std::string activation1,
    const int n2, const int k2, std::string activation2,
    const int n3, const int k3, std::string activation3,
    torch::Tensor weight1,  torch::Tensor weight2, torch::Tensor weight3,
    torch::Tensor mlp_in,  torch::Tensor output){

    using b2b_gemm = typename cutlass::gemm::device::FusedMultiGemmForward<cutlass::half_t>;

    cutlass::half_t alpha0 = cutlass::half_t(1);
    cutlass::half_t beta0 = cutlass::half_t(0);
    cutlass::gemm::GemmCoord problem_size_0(M, n1, k1);
    cutlass::half_t alpha1 = cutlass::half_t(1);
    cutlass::half_t beta1 = cutlass::half_t(0);
    cutlass::gemm::GemmCoord problem_size_1(M, n2, k2);
    cutlass::half_t alpha2 = cutlass::half_t(1);
    cutlass::half_t beta2 = cutlass::half_t(0);
    cutlass::gemm::GemmCoord problem_size_2(M, n3, k3);
    int Batch = 1;
    typename b2b_gemm::Arguments arguments{
        problem_size_0,
        problem_size_1,
        problem_size_2,
        {reinterpret_cast<cutlass::half_t*>(mlp_in.data_ptr<torch::Half>()), problem_size_0.k()},
        {reinterpret_cast<cutlass::half_t*>(weight1.data_ptr<torch::Half>()), k1},
        {},
        {reinterpret_cast<cutlass::half_t*>(weight2.data_ptr<torch::Half>()), k2},
        {},
        {reinterpret_cast<cutlass::half_t*>(weight3.data_ptr<torch::Half>()), k3},
        {},
        {reinterpret_cast<cutlass::half_t*>(output.data_ptr<torch::Half>()), problem_size_2.n()},
        { alpha0, beta0},
        { alpha1, beta1},
        { alpha2, beta2},
        Batch};

        b2b_gemm gemm_op;
        gemm_op.initialize(arguments);
        gemm_op();

}

void fp16_use_defined(
    bool in_rowmajor, bool out_rowmajor,
    const int m, const int n, const int k, std::string activation,
    torch::Tensor input, torch::Tensor weight, torch::Tensor output
)
{
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm = select_gemm(in_rowmajor, out_rowmajor, activation);
    cutlass::half_t* in_ptr = (cutlass::half_t*)input.data_ptr();
    cutlass::half_t* weight_ptr = (cutlass::half_t*)weight.data_ptr();
    cutlass::half_t* out_ptr = (cutlass::half_t*)output.data_ptr();

    checkKernelErrors((gemm(in_ptr, weight_ptr, out_ptr, m, n, k, std::nullopt)));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("fp16_use_defined", &fp16_use_defined, "Fp16, accept user defined layout and m n k size",
    py::arg("in_rowmajor"), py::arg("out_rowmajor"),
    py::arg("m"), py::arg("n"), py::arg("k"), py::arg("activation"),
    py::arg("input"), py::arg("weight"), py::arg("output"));

    m.def("Fused3mlp", &Fused3mlp, "gemm based on cutlass",  py::arg("is_in_rowmajor"),
    py::arg("is_layer1out_rowmajor"), py::arg("is_layer2out_rowmajor"), py::arg("is_output_rowmajor"),
    py::arg("M"), py::arg("n1"), py::arg("k1"), py::arg("activation1"),
    py::arg("n2"), py::arg("k2"), py::arg("activation2"),
    py::arg("n3"), py::arg("k3"), py::arg("activation3"),
    py::arg("weight1"), py::arg("weight2"), py::arg("weight3"),
    py::arg("mlp_in"),  py::arg("output"));
}
