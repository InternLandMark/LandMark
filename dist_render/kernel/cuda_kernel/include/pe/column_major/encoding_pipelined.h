#pragma once
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>
#include <assert.h>
#include <vector>
#include <torch/torch.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

////////////////////////////////////////////////////////////////////////////////
// Because G80 class hardware natively supports global memory operations
// only with data elements of 4, 8 and 16 bytes, if structure size
// exceeds 16 bytes, it can't be efficiently read or written,
// since more than one global memory non-coalescable load/store instructions
// will be generated, even if __align__ option is supplied.
// "Structure of arrays" storage strategy offers best performance
// in general case. See section 5.1.2 of the Programming Guide.
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(8) {
  half one, two,three, four;
}
half4;

# define IDX2C(i,j,rows)  ((j)*(rows)+(i))
# define IDX2R(i,j,columns)  ((i)*(columns)+(j))

__global__ void copy_row2column_2x(
  const int num_threads_col,
  const uint64_t num_elements, //所有执行的 threads的个数 行*列
  const int indata_width, const int outdata_height,
    half* indata, half* outdata){
  const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_index >= num_elements) return;

  const int i =  thread_index % num_threads_col;//输出的行
  const int j = thread_index / num_threads_col; //输入的列 = 输出的列/4

  half2 inputs_row0 = reinterpret_cast<half2*> (indata + IDX2R(2*i,2*j,indata_width))[0];
  half2 inputs_row1 = reinterpret_cast<half2*> (indata + IDX2R(2*i + 1,2*j,indata_width))[0];

  reinterpret_cast<half2*> (outdata + IDX2C(2*i, 2*j,outdata_height))[0] =
  make_half2(inputs_row0.x, inputs_row1.x);
  reinterpret_cast<half2*> (outdata + IDX2C(2*i, 2*j +1,outdata_height))[0] =
  make_half2(inputs_row0.y, inputs_row1.y);


}


__global__ void copy_shared_column_8x(
  const int num_threads_col,
  const uint64_t num_elements, //所有执行的 threads的个数 行*列
  const int indata_width, const int outdata_height,
  half* indata_ptr, half* outdata){
  uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_index >= num_elements) return;

  int i =  thread_index % num_threads_col;//输出的行
  int j = thread_index / num_threads_col; //输入的列 = 输出的列/8
  half indata = indata_ptr[j];

  reinterpret_cast<half2*> (outdata + IDX2C(8*i, j, outdata_height))[0] =
  make_half2(indata, indata);
  reinterpret_cast<half2*> (outdata + IDX2C(8*i+2,j, outdata_height))[0] =
  make_half2(indata, indata);
  reinterpret_cast<half2*> (outdata + IDX2C(8*i+4,j, outdata_height))[0] =
  make_half2(indata, indata);
  reinterpret_cast<half2*> (outdata + IDX2C(8*i+6,j, outdata_height))[0] =
  make_half2(indata, indata);
}

/*
move origin feature or viewdirs to tmp tensor and encoding
*/
__global__ void padding_to_zero_2x(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        half2* data){
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (thread_index >= num_elements) return;
        (data + thread_index)[0] = __floats2half2_rn(0.0f, 0.0f);
        }

__global__ void padding_to_zero(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        half* data){
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (thread_index >= num_elements) return;
        // printf("(data + thread_index)[0] %f \n", (data + thread_index)[0]);

        (data + thread_index)[0] = __float2half_rn(0.0f);

        }

__global__ void encoding_pipelined_column_major_movedata_freq2(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const uint64_t n_frequencies,
        const uint64_t col_encoding_offset,
        const uint64_t col_data_offset,
        const uint64_t row_offset,
        const int indata_width,
        const int indata_height,
        const int outdata_width,
        const int outdata_height, // height of pipelined output
        const int num_threads_col, // 处理一列数据所需线程
        const half* data_in,
        half* data_out)
{
  const int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

  if (thread_index >= num_elements) return;
  const int i_out =  thread_index % num_threads_col;//输出的行
  const int i_in =  thread_index % num_threads_col + row_offset;//输入的行
  const int j = thread_index / num_threads_col; //输入的列 = 输出的列/4
  const half input = data_in[j*indata_height + i_in];

  /*
  move original data to output
  */
 (data_out + (col_data_offset + j)*outdata_height + i_out)[0] = input;
//  printf("i_in %d, j %d, input %f \n", i_in, j, input);
  /*
    sin                          cos
    freq0 freq1 ...........      freq0 freq1 ...........
    =======================      =======================

    */


  const half input_freq1 = __hmul_rn(__float2half_rn(2.0f), input);

  half sin_freq0 = hsin(input);   half sin_freq1 = hsin(input_freq1);
  half cos_freq0 = hcos(input);   half cos_freq1 = hcos(input_freq1);


  data_out[j*2*outdata_height + i_out] = 	sin_freq0;
  data_out[(j*2+1)*outdata_height + i_out] = 	sin_freq1;
  data_out[(j*2 + indata_width*n_frequencies)*outdata_height + i_out] = 	cos_freq0;
  data_out[(j*2+1+ indata_width*n_frequencies)*outdata_height + i_out] = 	cos_freq1;
}


__global__ void encoding_pipelined_column_major_movedata_freq2_2x(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const uint64_t n_frequencies,
        const uint64_t col_encoding_offset,
        const uint64_t col_data_offset,
        const uint64_t row_offset,
        const int indata_width,
        const int indata_height,
        const int outdata_width,
        const int outdata_height, // height of pipelined output
        const int num_threads_col, // 处理一列数据所需线程
        const half* data_in,
        half2* data_out)
{
  const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;

  if (thread_index >= num_elements) return;
  const uint64_t i_out =  thread_index % num_threads_col;//输出的行/2
  const uint64_t j = thread_index / num_threads_col; //输入的列 = 输出的列/4
  half2 inputs_freq0;

  inputs_freq0.x = data_in[j*indata_height + i_out*2 + row_offset];
  inputs_freq0.y = data_in[j*indata_height + i_out*2 + row_offset+1];


  /*
  move original data to output
  */
 (data_out + (col_data_offset + j)*outdata_height/2 + i_out)[0] = inputs_freq0;
  /*
    sin                          cos
    freq0 freq1 ...........      freq0 freq1 ...........
    =======================      =======================

    */


  half2 inputs_freq1 = __hmul2_rn(__floats2half2_rn(2.0f,2.0f), inputs_freq0);
  // sin freq0
  (data_out +  (col_encoding_offset+2*j)*outdata_height/2 + i_out)[0] = h2sin(inputs_freq0);
  // sin freq1
  (data_out + (col_encoding_offset+j*2+1)*outdata_height/2 + i_out)[0] = h2sin(inputs_freq1);

  // cos freq0
  (data_out + (col_encoding_offset+j*2 + indata_width*n_frequencies)*outdata_height/2 + i_out)[0] = h2cos(inputs_freq0);
  // cos freq0
  (data_out + (col_encoding_offset+j*2+1+ indata_width*n_frequencies)*outdata_height/2 + i_out)[0] = h2cos(inputs_freq1);
}



__global__ void encoding_pipelined_column_major_kernel_freq2_2x(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const uint64_t n_frequencies,
        const uint64_t col_offset, // 30 for features
        const uint64_t row_offset,
        const int indata_width,
        const int indata_height,
        const int outdata_width,
        const int outdata_height, // height of pipelined output
        const int num_threads_col, // 处理一列数据所需线程
        const half* data_in,
        half2* data_out)
{
  const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;

  if (thread_index >= num_elements) return;
  const uint64_t i_out =  thread_index % num_threads_col;//输出的行/2
  const uint64_t j = thread_index / num_threads_col; //输入的列 = 输出的列/4
  /*
    sin                          cos
    freq0 freq1 ...........      freq0 freq1 ...........
    =======================      =======================

    */

  half2 inputs_freq0;
  inputs_freq0.x = data_in[j*indata_height + i_out*2 + row_offset];
  inputs_freq0.y = data_in[j*indata_height + i_out*2 + row_offset+1];

  half2 inputs_freq1 = __hmul2_rn(__floats2half2_rn(2.0f,2.0f), inputs_freq0);
  // sin freq0
  (data_out +  j*outdata_height + i_out)[0] = h2sin(inputs_freq0);
  // sin freq1
  (data_out + (j*2+1)*outdata_height/2 + i_out)[0] = h2sin(inputs_freq1);

  // cos freq0
  (data_out + (j*2 + indata_width*n_frequencies)*outdata_height/2 + i_out)[0] = h2cos(inputs_freq0);
  // cos freq0
  (data_out + (j*2+1+ indata_width*n_frequencies)*outdata_height/2 + i_out)[0] = h2cos(inputs_freq1);
}


__global__ void encoding_pipelined_column_major_kernel_freq2(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const uint64_t n_frequencies,
        const uint64_t col_offset, // 30 for features
        const uint64_t row_offset,
        const int indata_width,
        const int indata_height,
        const int outdata_width,
        const int outdata_height, // height of pipelined output
        const int num_threads_col, // 处理一列数据所需线程
        const half* data_in,
        half* data_out)
{
  const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;

  if (thread_index >= num_elements) return;
  const uint64_t i_out =  thread_index % num_threads_col;//输出的行
  const uint64_t i_in =  thread_index % num_threads_col + row_offset;//输入的行
  const uint64_t j_in = thread_index / num_threads_col; //输入的列 = 输出的列/4

  if (i_in > indata_height) { // padding to zero
    data_out[j_in*2*outdata_height + i_out] = __float2half_rn(0.0f);
    data_out[(j_in*2+1)*outdata_height + i_out] = __float2half_rn(0.0f);
    data_out[(j_in*2 + indata_width*n_frequencies)*outdata_height + i_out] = 	__float2half_rn(0.0f);
    data_out[(j_in*2+1+ indata_width*n_frequencies)*outdata_height + i_out] = 	__float2half_rn(0.0f);
    return;
  }
  /*
  sin                          cos
  freq0 freq1 ...........      freq0 freq1 ...........
  =======================      =======================

  */

  const half input = data_in[j_in*indata_height + i_in];
  const half input_freq1 = __hmul_rn(__float2half_rn(2.0f), input);

  half sin_freq0 = hsin(input);   half sin_freq1 = hsin(input_freq1);
  half cos_freq0 = hcos(input);   half cos_freq1 = hcos(input_freq1);


  data_out[j_in*2*outdata_height + i_out] = 	sin_freq0;
  data_out[(j_in*2+1)*outdata_height + i_out] = 	sin_freq1;
  data_out[(j_in*2 + indata_width*n_frequencies)*outdata_height + i_out] = 	cos_freq0;
  data_out[(j_in*2+1+ indata_width*n_frequencies)*outdata_height + i_out] = 	cos_freq1;
}
