#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <mma.h>

#include <ATen/AccumulateType.h>

#include <assert.h>
#include <vector>
#include <torch/torch.h>

# define blockDIM 128
# define PI 3.14159

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


__global__ void frequency_encoding_kernel(
        const uint32_t num_elements, //输出的个数 行*列
        const uint32_t n_frequencies,
        const uint32_t num_to_encode, //输入的宽度
        const uint32_t num_to_pad,
        const float*  data_in,
        float* data_out)
{
        const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;

        const uint32_t fan_out_encoded = num_to_encode * n_frequencies * 2;
        const uint32_t fan_out = fan_out_encoded + num_to_pad;

        const uint32_t i = encoded_index / fan_out; //行
        const uint32_t j = encoded_index - i * fan_out; //输出的列

        /* Layout of outputs (for each input record):
         *     frequency-encoded input dimension 0
         *     frequency-encoded input dimension 1
         *     frequency-encoded input dimension ...
         *     passthrough inputs
         *     padding (value 1.f)
         */
        if (j >= fan_out_encoded) {
                data_out[j+ i*fan_out] = 1;// padding 的列
        } else {

                // change layoyt
                const float phase_shift = (j / (num_to_encode * n_frequencies) ) * (PI/2);//sin ------  || cos-------
                const uint32_t log2_frequency = j  % n_frequencies; // freq0 freq1 ; freq0 freq1
                const uint32_t encoded_input_feature_i = (j %(num_to_encode * n_frequencies))/ n_frequencies;//输入的列

                // const uint32_t encoded_input_feature_i = j / (n_frequencies * 2);//输入的列
                // const uint32_t log2_frequency = (j / 2) % n_frequencies; //

                // const float phase_shift = (j % 2) * (PI/2);//sin cos sin cos
                const float x = scalbnf(data_in[encoded_input_feature_i + i*num_to_encode], log2_frequency);

                // const float x = scalbnf(data_in(encoded_input_feature_i, i), log2_frequency);
                const float input = x  + phase_shift;
                data_out[j+ i*fan_out] = (float)__sinf(input);

        }
}


torch::Tensor frequency_encoding(torch::Tensor features, int n_freque){

    const int indata_height = features.size(0);
    const int indata_width = features.size(1);

    torch::Tensor encoded_out = torch::zeros({indata_height, indata_width*2*n_freque}, features.options());
    int gridDIM =  (indata_height*indata_width*n_freque*2 + blockDIM-1)/blockDIM;

    AT_DISPATCH_FLOATING_TYPES(
         features.type(), "frequency_encoding_kernel", ([&] {
        frequency_encoding_kernel<<<gridDIM, blockDIM>>>
        (indata_height*indata_width*n_freque*2,
         n_freque, indata_width, 0, features.data<float>(),
        encoded_out.data<float>());
     }));

    getLastCudaError("requency_encoding execution failed\n");

    return encoded_out;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("frequency_encoding", &frequency_encoding, "frequency_encoding (CUDA)",    py::arg("features"), py::arg("n_freque"));
}
