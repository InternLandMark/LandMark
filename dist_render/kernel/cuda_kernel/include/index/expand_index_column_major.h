# pragma once
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

# define IDX2R(i,j,columns)  (j+(i*columns))



// 高度是单数
__device__ half expand_index_half2half(
        const uint32_t i, const uint32_t j,
        const uint32_t indata_width,
        const int in_offset,
        half* indata,
        int* index
        )
{
        // const uint64_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        // if (encoded_index >= num_elements) return;
        /*
        相邻的线程处理同一列
        */
        // const uint32_t i =  encoded_index % num_threads_col;//输出的行/2
        // const uint32_t j = encoded_index / num_threads_col; //输出的列

        int in_idx0 = index[i];

        return indata[IDX2R(in_idx0, j+in_offset, indata_width)];


}

// 高度是复数
__device__ half2 expand_index_half2half_2x(
        const uint32_t i, const uint32_t j,
        const uint32_t indata_width,
        const int in_offset,
        half* indata,
        int* index
        )
{
        // const uint64_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        // if (encoded_index >= num_elements) return;
        /*
        相邻的线程处理同一列
        */
        // const uint32_t i =  encoded_index % num_threads_col;//输出的行
        // const uint32_t j = encoded_index / num_threads_col; //输出的列

        int in_idx0 = index[2*i]; int in_idx1 = index[2*i + 1];

        return  make_half2(indata[IDX2R(in_idx0, j+in_offset, indata_width)],
        indata[IDX2R(in_idx1, j+in_offset, indata_width)]);

}
