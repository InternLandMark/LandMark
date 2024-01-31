#pragma once


#include "common/index.h"
#include "common/helper.h"

# define PI 3.14159

__global__ void frequency_encoding_simple_row_row(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const int n_frequencies,
        const int in_width,
        const int in_valid_width, // valid width of input data(width to be encoded))
        const int out_width, // height of pipelined output
        const int num_threads_row, // 处理一列数据所需线程
        const int in_off,
        const int out_off,
        half* data_in,
        half* data_out)
{
        const uint64_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;


        const uint32_t i = encoded_index / num_threads_row; //行
        const uint32_t j = encoded_index % num_threads_row; //输出的列


        const float phase_shift = (j / (in_valid_width * n_frequencies) ) * (PI/2);//sin ------  || cos-------
        const uint32_t log2_frequency = j  % n_frequencies; // freq0 freq1 ; freq0 freq1
        const uint32_t encoded_input_feature_i = (j %(in_valid_width * n_frequencies))/ n_frequencies;//输入的列
        const float x = scalbnf(__half2float(data_in[IDX2R(i, encoded_input_feature_i + in_off, in_width)]), log2_frequency);
        const float input = x  + phase_shift;
        data_out[IDX2R(i, j+out_off, out_width)] =  __float2half_rn(__sinf(input));

}





template <int n_freq>
__global__ void encoding_half2half_row_row_in1_out1(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const int in_width,
        const int in_valid_width, // valid width of input data(width to be encoded))
        const int out_width, // height of pipelined output
        const int num_threads_col, // 处理一列数据所需线程
        const int in_off,
        const int out_off,
        half* data_in,
        half* data_out)
{
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (thread_index >= num_elements) return;
        const uint64_t i_out =  thread_index / num_threads_col;//输出的行
        const uint64_t j = thread_index % num_threads_col; //输入的列 = 输出的列

        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================
        */

        half input_freq0 = data_in[IDX2R(i_out, j+in_off, in_width)];
        data_out[IDX2R(i_out, j*n_freq+out_off, out_width)] = hsin(input_freq0);
        data_out[IDX2R(i_out, j*n_freq + in_valid_width*n_freq+out_off, out_width)] = hcos(input_freq0);

        float alpha_freq = 2.0f;
        #pragma unroll
        for (int i = 1; i < n_freq; i++){
                half input_freqi = __hmul_rn(__float2half_rn(alpha_freq), input_freq0);
                data_out[IDX2R(i_out, j*n_freq+i+out_off, out_width)] = hsin(input_freqi);
                data_out[IDX2R(i_out, j*n_freq + in_valid_width*n_freq +i+out_off, out_width)] = hcos(input_freqi);

                alpha_freq *= 2;
        }

}

template <int n_freq>
__global__ void encoding_half2half_row_row_in1_out1_mv_data(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const int in_width,
        const int in_valid_width, // valid width of input data(width to be encoded))
        const int out_width, // height of pipelined output
        const int num_threads_col, // 处理一列数据所需线程
        const int in_read_off,
        const int in_write_off,
        const int out_off,
        half* data_in,
        half* data_out)
{
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (thread_index >= num_elements) return;
        const uint64_t i_out =  thread_index / num_threads_col;//输出的行
        const uint64_t j = thread_index % num_threads_col; //输入的列 = 输出的列

        half input_freq0 = data_in[IDX2R(i_out, j+in_read_off, in_width)];

        // write original indata into outdata
        data_out[IDX2R(i_out,  j+in_write_off, out_width)] = input_freq0;

        //encoding
        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================
        */
        data_out[IDX2R(i_out, j*n_freq+out_off, out_width)] = hsin(input_freq0);
        data_out[IDX2R(i_out, j*n_freq + in_valid_width*n_freq+out_off, out_width)] = hcos(input_freq0);

        float alpha_freq = 2.0f;
        #pragma unroll
        for (int i = 1; i < n_freq; i++){
                half input_freqi = __hmul_rn(__float2half_rn(alpha_freq), input_freq0);
                data_out[IDX2R(i_out, j*n_freq+i+out_off, out_width)] = hsin(input_freqi);
                data_out[IDX2R(i_out, j*n_freq + in_valid_width*n_freq +i+out_off, out_width)] = hcos(input_freqi);

                alpha_freq *= 2;
        }

}

__global__ void encoding_half2half_row_row_in1_out2_freq2_mv_data(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const uint64_t n_frequencies,
        const int in_width,
        const int in_valid_width, // valid width of input data(width to be encoded))
        const int out_width, // height of pipelined output
        const int num_threads_row, // 处理一列数据所需线程
        const int in_read_off,
        const int in_write_off,
        const int out_off,
        half* data_in,
        half* data_out)
{
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (thread_index >= num_elements) return;
        const uint64_t i_out =  thread_index / num_threads_row;//输出的行
        const uint64_t j = thread_index % num_threads_row; //输入的列 = 输出的列

        half input = data_in[IDX2R(i_out, j+in_read_off, in_width)];

        // write original indata into outdata
        data_out[IDX2R(i_out,  j+in_write_off, out_width)] = input;

        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================
        */
        half2 input_freqs = make_half2(input, __hmul_rn( __float2half_rn(2.0f), input)); // freq0 freq1
        reinterpret_cast<half2*>(data_out+ IDX2R(i_out, j*n_frequencies+out_off, out_width))[0]= h2sin(input_freqs);
        reinterpret_cast<half2*>(data_out+ IDX2R(i_out, j*n_frequencies+in_valid_width*n_frequencies+out_off, out_width))[0]= h2cos(input_freqs);
}


__global__ void encoding_half2half_row_row_in1_out2_freq2(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const uint64_t n_frequencies,
        const int in_width,
        const int in_valid_width, // valid width of input data(width to be encoded))
        const int out_width, // height of pipelined output
        const int num_threads_row, // 处理一列数据所需线程
        const int in_off,
        const int out_off,
        half* data_in,
        half* data_out)
{
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (thread_index >= num_elements) return;
        const uint64_t i_out =  thread_index / num_threads_row;//输出的行
        const uint64_t j = thread_index % num_threads_row; //输入的列 = 输出的列

        half input = data_in[IDX2R(i_out, j+in_off, in_width)];
        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================
        */
        half2 input_freqs = make_half2(input, __hmul_rn( __float2half_rn(2.0f), input)); // freq0 freq1
        reinterpret_cast<half2*>(data_out+ IDX2R(i_out, j*n_frequencies+out_off, out_width))[0]= h2sin(input_freqs);
        reinterpret_cast<half2*>(data_out+ IDX2R(i_out, j*n_frequencies+in_valid_width*n_frequencies+out_off, out_width))[0]= h2cos(input_freqs);
}



template <int n_freq>
__global__ void encoding_half2half_col_col_in2_out2(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const int indata_width, // valid width of input data(width to be encoded))
        const int data_height, // height of pipelined output
        const int num_threads_row, // 处理一列数据所需线程
        half2* data_in,
        half2* data_out)
{
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;

        if (thread_index >= num_elements) return;
        const uint64_t i_out =  thread_index % num_threads_row;//输出的行/2
        const uint64_t j = thread_index / num_threads_row; //输入的列 = 输出的列
        half2 inputs_freq0 = (data_in + IDX2C(i_out, j, data_height/2))[0];
        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================
        */


        // sin freq0
        (data_out + IDX2C(i_out, j*n_freq, data_height/2))[0] = h2sin(inputs_freq0);
        // cos freqi
        (data_out + IDX2C(i_out, j*n_freq + indata_width*n_freq, data_height/2))[0] = h2cos(inputs_freq0);

        float alpha_freq = 2.0f;
        #pragma unroll
        for (int i = 1; i < n_freq; i++){
                half2 inputs_freqi = __hmul2_rn(__floats2half2_rn(alpha_freq, alpha_freq), inputs_freq0);
                // sin freq1
                (data_out + IDX2C(i_out, j*n_freq +i, data_height/2))[0] = h2sin(inputs_freqi);
                // cos freqi
                (data_out + IDX2C(i_out, j*n_freq +i+ indata_width*n_freq, data_height/2))[0] = h2cos(inputs_freqi);
                alpha_freq *= 2;
        }

}

template <int n_freq>
__global__ void encoding_half2half_col_col_in1_out2(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const int indata_width, // valid width of input data(width to be encoded))
        const int data_height, // height of pipelined output
        const int num_threads_col, // 处理一列数据所需线程
        const half* data_in,
        half2* data_out)
{
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;

        if (thread_index >= num_elements) return;
        const uint64_t i_out =  thread_index % num_threads_col;
        const uint64_t j = thread_index / num_threads_col; //输入的列 = 输出的列
        half2 inputs_freq0;

        inputs_freq0.x = data_in[IDX2C(i_out*2, j, data_height)];
        inputs_freq0.y = data_in[IDX2C(i_out*2+1, j, data_height)];
        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================
        */
        // sin freq0
        (data_out + IDX2C(i_out, j*n_freq, data_height/2))[0] = h2sin(inputs_freq0);
        // cos freqi
        (data_out + IDX2C(i_out, j*n_freq + indata_width*n_freq, data_height/2))[0] = h2cos(inputs_freq0);

        float alpha_freq = 2.0f;
        #pragma unroll
        for (int i = 1; i < n_freq; i++){
                half2 inputs_freqi = __hmul2_rn(__floats2half2_rn(alpha_freq, alpha_freq), inputs_freq0);
                // sin freq1
                (data_out + IDX2C(i_out, j*n_freq +i, data_height/2))[0] = h2sin(inputs_freqi);
                // cos freqi
                (data_out + IDX2C(i_out, j*n_freq +i+ indata_width*n_freq, data_height/2))[0] = h2cos(inputs_freqi);
                alpha_freq *= 2;
        }
}

template <int n_freq>
__global__ void encoding_half2half_col_col_in1_out1(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const int indata_width, // valid width of input data(width to be encoded))
        const int data_height, // height of pipelined output
        const int num_threads_col, // 处理一列数据所需线程
        const half* data_in,
        half* data_out)
{
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;

        if (thread_index >= num_elements) return;
        const uint64_t i_out =  thread_index % num_threads_col;
        const uint64_t j = thread_index / num_threads_col; //输入的列 = 输出的列
        const half input_freq0 = data_in[IDX2C(i_out, j, data_height)];
        data_out[IDX2C(i_out, j*n_freq, data_height)] = hsin(input_freq0);
        data_out[IDX2C(i_out, j*n_freq+ indata_width*n_freq, data_height)] = hcos(input_freq0);
        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================
        */

       float alpha_freq = 2.0f;
        #pragma unroll
        for (int i = 1; i < n_freq; i++){
                half input_freqi = __hmul_rn(__float2half_rn(alpha_freq), input_freq0);
                // sin freqi
                data_out[IDX2C(i_out, j*n_freq+i, data_height)] = hsin(input_freqi);
                // cos freqi
                data_out[IDX2C(i_out, j*n_freq + indata_width*n_freq +i, data_height)] = hcos(input_freqi);
                alpha_freq *= 2;
        }
}
// typedef struct __align__(8) {
//   half x, y, z, k;
// }
// half4;


typedef struct __align__(8) {
  half2 x, y, z, k;
}
half8;

typedef struct __align__(8) {
  half2 x, y;
}
half4;

template <class TData, int alignment, int n_freq>
__global__ void encoding_half2half_col_col_in4_out4(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const int indata_width, // valid width of input data(width to be encoded))
        const int data_height, // height of pipelined output
        const int num_threads_row, // 处理一列数据所需线程
        TData* data_in,
        TData* data_out)
{
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;

        if (thread_index >= num_elements) return;
        const uint64_t i_out =  thread_index % num_threads_row;//输出的行/2
        const uint64_t j = thread_index / num_threads_row; //输入的列 = 输出的列
        TData inputs_freq0 = (data_in + IDX2C(i_out, j, data_height/alignment))[0];
        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================
        */
        TData encoded_data_sin; TData encoded_data_cos;

        // for freq0
        encoded_data_sin.x = h2sin(inputs_freq0.x);
        encoded_data_sin.y = h2sin(inputs_freq0.y);
        encoded_data_cos.x = h2cos(inputs_freq0.x);
        encoded_data_cos.y = h2cos(inputs_freq0.y);

        (data_out + IDX2C(i_out, j*n_freq, data_height/alignment))[0] = encoded_data_sin;
        (data_out + IDX2C(i_out, j*n_freq+indata_width*n_freq, data_height/alignment))[0] = encoded_data_cos;


        float alpha_freq = 2.0f; half8 inputs_freqi; half2 alpha_freqs;
        #pragma unroll
        for (int i = 1; i < n_freq; i++){
                alpha_freqs = __floats2half2_rn(alpha_freq, alpha_freq);
                inputs_freqi.x = __hmul2_rn(alpha_freqs, inputs_freq0.x);
                inputs_freqi.y = __hmul2_rn(alpha_freqs, inputs_freq0.y);


                encoded_data_sin.x = h2sin(inputs_freqi.x);
                encoded_data_sin.y = h2sin(inputs_freqi.y);


                encoded_data_cos.x = h2cos(inputs_freqi.x);
                encoded_data_cos.y = h2cos(inputs_freqi.y);


                (data_out + IDX2C(i_out, j*n_freq +i, data_height/alignment))[0] = encoded_data_sin;
                (data_out + IDX2C(i_out, j*n_freq +i+ indata_width*n_freq, data_height/alignment))[0] = encoded_data_cos;

                alpha_freq *= 2;
        }

}

template <class TData, int alignment, int n_freq>
__global__ void encoding_half2half_col_col_in8_out8(
        const uint64_t num_elements, //所有执行的 threads的个数 行*列
        const int indata_width, // valid width of input data(width to be encoded))
        const int data_height, // height of pipelined output
        const int num_threads_row, // 处理一列数据所需线程
        TData* data_in,
        TData* data_out)
{
        const uint64_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;

        if (thread_index >= num_elements) return;
        const uint64_t i_out =  thread_index % num_threads_row;//输出的行/2
        const uint64_t j = thread_index / num_threads_row; //输入的列 = 输出的列
        TData inputs_freq0 = (data_in + IDX2C(i_out, j, data_height/alignment))[0];
        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================
        */
        TData encoded_data_sin; TData encoded_data_cos;

        // for freq0
        encoded_data_sin.x = h2sin(inputs_freq0.x);
        encoded_data_sin.y = h2sin(inputs_freq0.y);
        encoded_data_sin.z = h2sin(inputs_freq0.z);
        encoded_data_sin.k = h2sin(inputs_freq0.k);

        encoded_data_cos.x = h2cos(inputs_freq0.x);
        encoded_data_cos.y = h2cos(inputs_freq0.y);
        encoded_data_cos.z = h2cos(inputs_freq0.z);
        encoded_data_cos.k = h2cos(inputs_freq0.k);


        (data_out + IDX2C(i_out, j*n_freq, data_height/alignment))[0] = encoded_data_sin;
        (data_out + IDX2C(i_out, j*n_freq+indata_width*n_freq, data_height/alignment))[0] = encoded_data_cos;


        float alpha_freq = 2.0f; half8 inputs_freqi; half2 alpha_freqs;
        #pragma unroll
        for (int i = 1; i < n_freq; i++){
                alpha_freqs = __floats2half2_rn(alpha_freq, alpha_freq);
                inputs_freqi.x = __hmul2_rn(alpha_freqs, inputs_freq0.x);
                inputs_freqi.y = __hmul2_rn(alpha_freqs, inputs_freq0.y);
                inputs_freqi.z = __hmul2_rn(alpha_freqs, inputs_freq0.z);
                inputs_freqi.k = __hmul2_rn(alpha_freqs, inputs_freq0.k);

                encoded_data_sin.x = h2sin(inputs_freqi.x);
                encoded_data_sin.y = h2sin(inputs_freqi.y);
                encoded_data_sin.z = h2sin(inputs_freqi.z);
                encoded_data_sin.k = h2sin(inputs_freqi.k);

                encoded_data_cos.x = h2cos(inputs_freqi.x);
                encoded_data_cos.y = h2cos(inputs_freqi.y);
                encoded_data_cos.z = h2cos(inputs_freqi.z);
                encoded_data_cos.k = h2cos(inputs_freqi.k);

                (data_out + IDX2C(i_out, j*n_freq +i, data_height/alignment))[0] = encoded_data_sin;
                (data_out + IDX2C(i_out, j*n_freq +i+ indata_width*n_freq, data_height/alignment))[0] = encoded_data_cos;

                alpha_freq *= 2;
        }

}
