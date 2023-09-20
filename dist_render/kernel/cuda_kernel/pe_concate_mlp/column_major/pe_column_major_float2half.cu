#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <mma.h>

#include <ATen/AccumulateType.h>

#include <assert.h>
#include <vector>
#include <torch/torch.h>

# define blockDIM 270
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
        const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t n_frequencies,
        const uint32_t offset, // 30 for features
        const int indata_width,
        const int indata_height,
        const int outdata_width,
        const int outdata_height,
        const int num_threads_col, // 处理一列数据所需线程
        half*  data_in,
        half* data_out)
{

        const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;

        const uint32_t i =  encoded_index % num_threads_col;//输出的行
        const uint32_t j = encoded_index / num_threads_col; //输出的列

        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================

        */

       const half phase_shift = __float2half_rn(j / (indata_width * n_frequencies)  *  (PI/2));//sin ------  || cos-------
       const uint32_t log2_frequency = j  % n_frequencies; // freq0 freq1 ; freq0 freq1
       const uint32_t encoded_input_feature_j = (j %(indata_width * n_frequencies))/ n_frequencies;//输入的列

      const half input = data_in[encoded_input_feature_j*outdata_height + i];
      const half x = 	__hmul(input, hexp2(__float2half_rn(float(log2_frequency)))); // input * 2^(log2_frequency)
      data_out[encoded_index + offset*outdata_height] = 	hsin(	__hadd(x,phase_shift));
      // data_out[encoded_index + offset*outdata_height] = data_in[encoded_input_feature_j*outdata_height + i];

  }

__global__ void frequency_encoding_kernel_2x(
        const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t n_frequencies,
        const uint32_t offset, // 30 for features
        const int indata_width,
        const int indata_height,
        const int outdata_width,
        const int outdata_height,
        const int num_threads_col, // 处理一列数据所需线程
        half*  data_in,
        half* data_out)
{

        const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;

        const uint32_t i =  encoded_index % num_threads_col;//输出的行/2
        const uint32_t j = encoded_index / num_threads_col; //输出的列

        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================

        */

      half phase_shift = __float2half_rn(j / (indata_width * n_frequencies)  *  (PI/2));//sin ------  || cos-------
      half2 phase_shifts = make_half2(phase_shift, phase_shift);
      half exp2_frequency = hexp2(__float2half_rn(float(j  % n_frequencies))); // freq0 freq1 ; freq0 freq1  2^(log2_frequency)
      uint32_t encoded_input_feature_j = (j %(indata_width * n_frequencies))/ n_frequencies;//输入的列

      half2 inputs = reinterpret_cast<half2*>(data_in+encoded_input_feature_j*outdata_height + i*2)[0];
      half2 x = 	__hmul2_rn(inputs,
      make_half2(exp2_frequency, exp2_frequency)); // input * 2^(log2_frequency)
      reinterpret_cast<half2*>(data_out + encoded_index*2 + offset*outdata_height)[0] = 	h2sin(	__hadd2_rn (x,phase_shifts));
      // data_out[encoded_index + offset*outdata_height] = data_in[encoded_input_feature_j*outdata_height + i];

  }

__global__ void frequency_encoding_kernel_2x_freq2
(const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t n_frequencies,
        const uint32_t offset, // 30 for features
        const int indata_width,
        const int indata_height,
        const int outdata_width,
        const int outdata_height,
        const int num_threads_col, // 处理一列数据所需线程
        half*  data_in,
        half* data_out)
{

        const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;

        const uint32_t i =  encoded_index % num_threads_col;//输出的行/2 = 输如的行/2
        const uint32_t j = encoded_index / num_threads_col; //输入的列 = 输出的列/4

        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================

        */

       half2 inputs_freq0 = reinterpret_cast<half2*>(data_in+ j*outdata_height + i*2)[0];
      half2 inputs_freq1 = __hmul2_rn(__floats2half2_rn(2.0f,2.0f), inputs_freq0);
       // sin freq0
       reinterpret_cast<half2*>(data_out +  j*2*outdata_height + i*2)[0] = h2sin(inputs_freq0);
       // sin freq1
       reinterpret_cast<half2*>(data_out + ( j*2+1)*outdata_height + i*2)[0] = h2sin(inputs_freq1);

       // cos freq0
       reinterpret_cast<half2*>(data_out + (j*2 + indata_width*n_frequencies)*outdata_height + i*2)[0] = h2cos(inputs_freq0);
       // cos freq0
       reinterpret_cast<half2*>(data_out + (j*2+1+ indata_width*n_frequencies)*outdata_height + i*2)[0] = h2cos(inputs_freq1);



  }

void frequency_encoding(torch::Tensor indata, torch::Tensor outdata, int n_freque, int offset){

    /*
    indata shape N1 *3
    outdata shape N2 * 152
    index shape N2
    */
    const int indata_height = indata.size(0);
    const int outdata_height = outdata.size(0);
    const int indata_width = indata.size(1);
    const int outdata_width = outdata.size(1);


   if (outdata_height%2 == 0){

      std::cout << "vectorized access half2" << std::endl;
      if (n_freque == 2){


        const int num_threads_col = outdata_height/2;
        const int num_elements = (indata_width)*num_threads_col;
        int gridDIM = (num_elements + blockDIM-1)/blockDIM;

        std::cout << "Specialized kernel for freq = 2" << std::endl;
        frequency_encoding_kernel_2x_freq2<<<gridDIM, blockDIM>>>
          (num_elements,
            n_freque,
            offset,
            indata_width,
            indata_height,
            outdata_width,
            outdata_height,
            num_threads_col,
            reinterpret_cast<half*> (indata.to(at::kHalf).data_ptr<torch::Half>()),
            reinterpret_cast<half*> (outdata.data_ptr<torch::Half>() + offset*outdata_height)
          );
      }
      else{
        const int num_threads_col = outdata_height/2;
        const int num_elements = (indata_width*n_freque*2)*num_threads_col;
        int gridDIM = (num_elements + blockDIM-1)/blockDIM;



        frequency_encoding_kernel_2x<<<gridDIM, blockDIM>>>
        (num_elements,
          n_freque,
          offset,
          indata_width,
          indata_height,
          outdata_width,
          outdata_height,
          num_threads_col,
          reinterpret_cast<half*> (indata.to(at::kHalf).data_ptr<torch::Half>()),
          reinterpret_cast<half*> (outdata.data_ptr<torch::Half>())
        );
       }
    }
    else{
      const int num_threads_col = outdata_height;
      const int num_elements = (indata_width*n_freque*2)*num_threads_col;
      int gridDIM = (num_elements + blockDIM-1)/blockDIM;
      frequency_encoding_kernel<<<gridDIM, blockDIM>>>
        (num_elements,
          n_freque,
          offset,
          indata_width,
          indata_height,
          outdata_width,
          outdata_height,
          num_threads_col,
          reinterpret_cast<half*> (indata.to(at::kHalf).data_ptr<torch::Half>()),
          reinterpret_cast<half*> (outdata.data_ptr<torch::Half>())
        );
    }


    getLastCudaError("requency_encoding execution failed\n");

    return;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &frequency_encoding, "frequency_encoding (CUDA)");
}
