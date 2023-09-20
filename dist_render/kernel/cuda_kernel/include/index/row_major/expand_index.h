#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>



#include "common/index.h"





__device__ void expand_index_half2half(
        const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t indata_height,
        const uint32_t outdata_height,
        const uint32_t indata_width,
        const uint32_t outdata_width,
        const int num_threads_row,
        const int offset,
        half* indata,
        int* index,
        half* outdata
        )
{
        const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;

        const uint32_t i = encoded_index / num_threads_row; //输出的行
        const uint32_t j = encoded_index - i * num_threads_row; //输出的列

        int in_idx = index[i];

        outdata[i*outdata_width + j + offset] = indata[in_idx*indata_width + j];

}
