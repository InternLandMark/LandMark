#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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


__global__ void pe_concate_kernel (
        const uint64_t num_elements, //输出的个数 行*列
        const uint32_t output_width, //输出的宽度
        const uint32_t input1_width,
        const uint32_t input2_width,
        const uint32_t n_freq1,
        const uint32_t n_freq2,
        const float*  data_in1,
        const float*  data_in2,
        half* data_out)
{
        const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;


        const uint32_t i = encoded_index / output_width; //行
        const uint32_t j = encoded_index - i * output_width; //输出的列
        uint32_t input_j;//输入的列
        uint32_t encoding_j;//encoding 的列

        if (j<input1_width){ // fill feature data
            data_out[j+ i*output_width] = __float2half_rn(data_in1[j + i*input1_width]);
        }
        else if (j<input1_width + input2_width){ // fill viewdir data
            input_j = j-input1_width;
            data_out[j+ i*output_width] = __float2half_rn(data_in2[input_j + i*input2_width]);
        }

        else if (j<input1_width + input2_width + input1_width * n_freq1*2){ // encoding
            encoding_j = j-input1_width-input2_width;
            const float phase_shift = (encoding_j / (input1_width * n_freq1) ) * (PI/2);//sin ------  || cos-------
            const uint32_t log2_frequency = encoding_j  % n_freq1; // freq0 freq1 ; freq0 freq1
            const uint32_t encoded_input_feature_i = (encoding_j %(input1_width * n_freq1))/ n_freq1;//输入的列

            const float x = scalbnf(data_in1[encoded_input_feature_i + i*input1_width], log2_frequency);

            const float input = x  + phase_shift;
            data_out[j+ i*output_width] = __float2half_rn(__sinf(input));
        }

        else if (j<input1_width + input2_width + input1_width * n_freq1*2 +  input2_width * n_freq2*2){ // encoding
            encoding_j = j-input1_width-input2_width-input1_width * n_freq2*2;
            const float phase_shift = (encoding_j / (input2_width * n_freq2) ) * (PI/2);//sin ------  || cos-------
            const uint32_t log2_frequency = encoding_j  % n_freq2; // freq0 freq1 ; freq0 freq1
            const uint32_t encoded_input_feature_i = (encoding_j %(input2_width * n_freq2))/ n_freq2;//输入的列

            const float x = scalbnf(data_in2[encoded_input_feature_i + i*input2_width], log2_frequency);

            const float input = x  + phase_shift;
            data_out[j+ i*output_width] = __float2half_rn(__sinf(input));
        }
        else data_out[j+ i*output_width] =  __float2half_rn((float)0.0);
}


void pe_concate(torch::Tensor data_in1,  int n_freq1, torch::Tensor data_in2, int n_freq2,  torch::Tensor data_out){

    const int indata_height = data_in1.size(0);
    const int indata_width1 = data_in1.size(1);
    const int indata_width2 = data_in2.size(1);
    const int non_padding_width = indata_width1 + indata_width2 + indata_width1*n_freq1*2 + indata_width2*n_freq2*2;
    const int output_width = data_out.size(1); // width ofthe data_out after padding (152)
    if (output_width<non_padding_width){
      std::cerr << "Error: output width " << output_width <<" should be >= non padding width " << non_padding_width << std::endl;
    }
    const uint64_t num_elements = indata_height * output_width;
    const uint32_t gridDIM = (num_elements + blockDIM-1)/blockDIM;


    pe_concate_kernel<<<gridDIM, blockDIM>>>
    (
        num_elements, //输出的个数 行*列
        output_width, //输出的宽度
        indata_width1,
        indata_width2,
        n_freq1,
        n_freq2,
        data_in1.data<float>(),
        data_in2.data<float>(),
        reinterpret_cast<half*> (data_out.data_ptr<torch::Half>())
      );


    getLastCudaError("pe_concate execution failed\n");

    return;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pe_concate", &pe_concate, "pe_concate (CUDA)",
  py::arg("data_in1"), py::arg("n_freq1"), py::arg("data_in2"), py::arg("n_freq2"), py::arg("data_out"));
}
