#pragma once
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <mma.h>

#include <ATen/AccumulateType.h>

#include <assert.h>
#include <vector>
#include <torch/torch.h>


#include "common/index.h"
# define PI 3.14159



/*
non vectorized read
vectorized(half2) write
*/
__device__ void frequency_encoding_1read2write(
        const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t n_frequencies,
        const int indata_width,
        const int num_threads_row, // 处理一行数据所需线程
        const uint32_t offset, // 30 for features
        const uint32_t dataout_width, // 152
        half*  data_in,
        half* data_out)
{

        const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;

        const uint32_t i = encoded_index / num_threads_row; //输入的行
        const uint32_t j = encoded_index - i * num_threads_row; //输入的列

        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================

        */

        half input = data_in[i*indata_width + j];
        half2 input_freqs = make_half2(input, __hmul_rn( __float2half_rn(2.0f), input)); // freq0 freq1
        half2 out_sin; half2 out_cos;
        out_sin = h2sin(input_freqs);
        out_cos = h2cos(input_freqs);

        reinterpret_cast<half2*>(data_out+ i*dataout_width + offset + j*n_frequencies)[0]= out_sin;
        reinterpret_cast<half2*>(data_out+ i*dataout_width + offset + j*n_frequencies + indata_width*n_frequencies)[0]= out_cos;

  }
