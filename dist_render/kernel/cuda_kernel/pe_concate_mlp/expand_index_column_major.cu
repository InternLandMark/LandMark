#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <mma.h>

#include <ATen/AccumulateType.h>

#include <assert.h>
#include <vector>
#include <torch/torch.h>

# define blockDIM 256
# define PI 3.1415926535

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



// 高度是单数
__global__ void expand_index_kernel_half2half(
        const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t indata_height,
        const uint32_t outdata_height,
        const uint32_t indata_width,
        const uint32_t outdata_width,
        const int num_threads_col,
        const int offset,
        half* indata,
        int* index,
        half* outdata
        )
{
        const uint64_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;
        /*
        相邻的线程处理同一列
        */
        const uint32_t i =  encoded_index % num_threads_col;//输出的行/2
        const uint32_t j = encoded_index / num_threads_col; //输出的列

        int in_idx0 = index[i];
        outdata[(offset + j) *outdata_height + i] = indata[j*indata_height + in_idx0];

}

// 高度是复数
__global__ void expand_index_kernel_half2half_2x(
        const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t indata_height,
        const uint32_t outdata_height,
        const uint32_t indata_width,
        const uint32_t outdata_width,
        const int num_threads_col,
        const int offset,
        half* indata,
        int* index,
        half* outdata
        )
{
        const uint64_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;
        /*
        相邻的线程处理同一列
        */
        const uint32_t i =  encoded_index % num_threads_col;//输出的行
        const uint32_t j = encoded_index / num_threads_col; //输出的列

        int in_idx0 = index[2*i]; int in_idx1 = index[2*i + 1];

        reinterpret_cast<half2*> (outdata + (offset + j) *outdata_height + 2*i) [0] =
        make_half2(indata[j*indata_height + in_idx0], indata[j*indata_height + in_idx1]);

}


void expand_index(torch::Tensor indata, torch::Tensor index, torch::Tensor outdata, int offset){
        /*
        indata shape N1 *3
        outdata shape N2 * 153
        index shape N2
        */
        const int indata_height = indata.size(0);
        const int outdata_height = outdata.size(0);
        const int indata_width = indata.size(1);
        const int outdata_width = outdata.size(1);

        if (outdata_height%2 == 0){
             const int num_threads_col = outdata_height/2;
                const int num_elements = indata_width*num_threads_col;
                int gridDIM = (num_elements + blockDIM-1)/blockDIM;

                if (indata.scalar_type() == torch::kFloat && indata.scalar_type() == torch::kHalf){
                        expand_index_kernel_half2half_2x<<<gridDIM, blockDIM>>>
                                (num_elements,
                                indata_height,
                                outdata_height,
                                indata_width,
                                outdata_width,
                                num_threads_col,
                                offset,
                                reinterpret_cast<half*> (indata.to(at::kHalf).data_ptr<torch::Half>()),
                                index.data_ptr<int>(),
                                reinterpret_cast<half*> (outdata.data_ptr<torch::Half>())
                        );
                }

                else if (indata.scalar_type() == torch::kHalf && indata.scalar_type() == torch::kHalf){
                        expand_index_kernel_half2half_2x<<<gridDIM, blockDIM>>>
                                (num_elements,
                                indata_height,
                                outdata_height,
                                indata_width,
                                outdata_width,
                                num_threads_col,
                                offset,
                                reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
                                index.data_ptr<int>(),
                                reinterpret_cast<half*> (outdata.data_ptr<torch::Half>())
                        );
                }
                else {
                std::cerr << "Input and output data type is not supported."  << std::endl;
                }
        }
        else{
                // std::cerr << "Only supports heights that are multiples of 2, but get"  << std::endl;
                const int num_threads_col = outdata_height;
                const int num_elements = indata_width*num_threads_col;
                int gridDIM = (num_elements + blockDIM-1)/blockDIM;

                if (indata.scalar_type() == torch::kFloat && indata.scalar_type() == torch::kHalf){
                        expand_index_kernel_half2half<<<gridDIM, blockDIM>>>
                                (num_elements,
                                indata_height,
                                outdata_height,
                                indata_width,
                                outdata_width,
                                num_threads_col,
                                offset,
                                reinterpret_cast<half*> (indata.to(at::kHalf).data_ptr<torch::Half>()),
                                index.data_ptr<int>(),
                                reinterpret_cast<half*> (outdata.data_ptr<torch::Half>())
                        );
                }

                else if (indata.scalar_type() == torch::kHalf && indata.scalar_type() == torch::kHalf){
                        expand_index_kernel_half2half<<<gridDIM, blockDIM>>>
                                (num_elements,
                                indata_height,
                                outdata_height,
                                indata_width,
                                outdata_width,
                                num_threads_col,
                                offset,
                                reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
                                index.data_ptr<int>(),
                                reinterpret_cast<half*> (outdata.data_ptr<torch::Half>())
                        );
                }
                else {
                std::cerr << "Input and output data type is not supported."  << std::endl;
                }
        }

        getLastCudaError("expand_index execution failed\n");

        return;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &expand_index, "expand_index (CUDA)");
}
