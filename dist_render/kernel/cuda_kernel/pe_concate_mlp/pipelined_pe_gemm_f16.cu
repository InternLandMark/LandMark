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
#include "helper.h"
# define blockDIM 256

#include "pe/column_major/encoding_pipelined.h"
#include "gemm/fp16.h"

# define debug 0
// #define STREAM_COUNT 2
// #define TMP_HEIGHT 108*256
// #define TMP_WIDTH 152

// #define GENERATE_TENSOR_VECTOR(STREAM_COUNT, TMP_HEIGHT, TMP_WIDTH) \
//     std::vector<torch::Tensor> tmp_mlp_in(STREAM_COUNT); \
//     for (int i = 0; i < STREAM_COUNT; ++i) { \
//         tmp_mlp_in[i] = torch::zeros({TMP_HEIGHT, TMP_WIDTH}, torch::kHalf).t().contiguous().t().cuda(); \
//     }

// #define CREATE_CUDA_STREAMS(STREAM_COUNT) \
//     cudaStream_t stream[STREAM_COUNT]; \
//     for (int i = 0; i < STREAM_COUNT; ++i) { \
//         cudaStreamCreate(&stream[i]); \
//     }

void encoding_gemm_fp16(torch::Tensor features, torch::Tensor viewdirs, int n_freque_fea, int n_freque_view,
int fea_encoding_offset, int view_encoding_offset, int fea_data_offset, int view_data_offset,
torch::Tensor Gb0,  torch::Tensor Gd0, torch::Tensor tmp_mlp_in,
 int padding_offset, int padding_width) {

    // cudaStream_t stream[stream_count];
    // for (int i = 0; i < stream_count; ++i) {
    //     cudaStreamCreate(&stream[i]);
    // }
    // GENERATE_TENSOR_VECTOR(STREAM_COUNT, TMP_HEIGHT, TMP_WIDTH);


    const int in_height = features.size(0);
    const int in_width_fea = features.size(1);
    const int in_width_view = viewdirs.size(1);

    const int tmp_height = tmp_mlp_in.size(0);
    const int tmp_width = tmp_mlp_in.size(1);


    int num_threads_col = tmp_height/2;

    int num_elements_fea = (in_width_fea)*num_threads_col;
    int num_elements_view = (in_width_view)*num_threads_col;
    int gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    int gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;

    const int n = Gb0.size(0); const int k = Gb0.size(1);

    half* output_begin = reinterpret_cast<half*> (Gd0.data_ptr<torch::Half>());


    const int iterations = (in_height + tmp_height -1 )/tmp_height;
    if (debug){
      printf("iterations %d \n", iterations);
    }
    cudaStream_t stream[3];
    for (int i = 0; i < 3; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    int final_height = in_height % tmp_height;
    if (final_height == 0){
        for (int i = 0; i < iterations; i++){

            int row_offset = i*tmp_height;
            // encoding for feature
            encoding_pipelined_column_major_movedata_freq2_2x<<<gridDIM_fea, blockDIM, 0, stream[0]>>>
            (
            num_elements_fea, //所有执行的 threads的个数 行*列
            n_freque_fea,
            fea_encoding_offset, // 30 for features
            fea_data_offset,
            row_offset,
            in_width_fea,
            in_height,
            tmp_width,
            tmp_height,
            num_threads_col, // 处理一列数据所需线程
            reinterpret_cast<half*> (features.data_ptr<torch::Half>()),
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
            );

            // encoding for viewdirs
            encoding_pipelined_column_major_movedata_freq2_2x<<<gridDIM_view, blockDIM, 0, stream[1]>>>
            (
            num_elements_view, //所有执行的 threads的个数 行*列
            n_freque_view,
            view_encoding_offset, // 30 for features
            view_data_offset,
            row_offset,
            in_width_view,
            in_height,
            tmp_width,
            tmp_height,
            num_threads_col, // 处理一列数据所需线程
            reinterpret_cast<half*> (viewdirs.data_ptr<torch::Half>()),
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
            );
            cudaDeviceSynchronize();

            // gemm_fp16_col_col_col_relu_4x_multiStreams(tmp_mlp_in[i%STREAM_COUNT], Gb0,  output_begin+ n*row_offset, stream[i%STREAM_COUNT]);
            gemm_fp16_col_col_col_relu_4x_problemsize(tmp_mlp_in, Gb0,  output_begin+ n*row_offset, tmp_height, n, k);
        }
    }
    else{
        int row_offset;
        for (int i = 0; i < iterations-1; i++)
        {

            row_offset = i*tmp_height;
            // encoding for feature
            encoding_pipelined_column_major_movedata_freq2_2x<<<gridDIM_fea, blockDIM, 0, stream[0]>>>
            (
            num_elements_fea, //所有执行的 threads的个数 行*列
            n_freque_fea,
            fea_encoding_offset, // 30 for features
            fea_data_offset,
            row_offset,
            in_width_fea,
            in_height,
            tmp_width,
            tmp_height,
            num_threads_col, // 处理一列数据所需线程
            reinterpret_cast<half*> (features.data_ptr<torch::Half>()),
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
            );

            // encoding for viewdirs
            encoding_pipelined_column_major_movedata_freq2_2x<<<gridDIM_view, blockDIM, 0, stream[1]>>>
            (
            num_elements_view, //所有执行的 threads的个数 行*列
            n_freque_view,
            view_encoding_offset, // 30 for features
            view_data_offset,
            row_offset,
            in_width_view,
            in_height,
            tmp_width,
            tmp_height,
            num_threads_col, // 处理一列数据所需线程
            reinterpret_cast<half*> (viewdirs.data_ptr<torch::Half>()),
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
            );
            cudaDeviceSynchronize();

            gemm_fp16_col_col_col_relu_4x_problemsize(tmp_mlp_in, Gb0,  output_begin+ n*row_offset, tmp_height, n, k);
        }
        /*
        For the last stage, if the corresponding row is a multiple of 8, we calculate it directly, otherwise, we pad upwards to 8 before doing the calculation.
        */

        row_offset = (final_height % 8) == 0? (iterations-1)*tmp_height : (iterations-1)*tmp_height-(8-(final_height % 8));
        int finalstage_height = in_height - row_offset;

        num_threads_col = finalstage_height/2;
        num_elements_fea = (in_width_fea)*num_threads_col;
        num_elements_view = (in_width_view)*num_threads_col;
        gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
        gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;
        const int num_padding = num_threads_col*padding_width;
        const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;

        if (debug){
        printf("row_offset %d \n", row_offset);
        printf("finalstage_height %d \n", finalstage_height);
        printf("num_padding %d \n", num_padding);
        }

        // encoding for feature
        encoding_pipelined_column_major_movedata_freq2_2x<<<gridDIM_fea, blockDIM, 0, stream[0]>>>
        (
        num_elements_fea, //所有执行的 threads的个数 行*列
        n_freque_fea,
        fea_encoding_offset, // 30 for features
        fea_data_offset,
        row_offset,
        in_width_fea,
        in_height,
        tmp_width,
        finalstage_height,
        num_threads_col, // 处理一列数据所需线程
        reinterpret_cast<half*> (features.data_ptr<torch::Half>()),
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
        );

        // encoding for viewdirs
        encoding_pipelined_column_major_movedata_freq2_2x<<<gridDIM_view, blockDIM, 0, stream[1]>>>
        (
        num_elements_view, //所有执行的 threads的个数 行*列
        n_freque_view,
        view_encoding_offset, // 30 for features
        view_data_offset,
        row_offset,
        in_width_view,
        in_height,
        tmp_width,
        finalstage_height,
        num_threads_col, // 处理一列数据所需线程
        reinterpret_cast<half*> (viewdirs.data_ptr<torch::Half>()),
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
        );

        padding_to_zero_2x<<<gridDIM_padding, blockDIM,  0, stream[2]>>>(
            num_padding,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*padding_offset)
        );
        cudaDeviceSynchronize();
        gemm_fp16_col_col_col_relu_4x_problemsize(tmp_mlp_in, Gb0,  output_begin+ n*row_offset, finalstage_height, n, k);
    }


}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &encoding_gemm_fp16, "gemm based on cutlass");
    //"Gpu_Cublas"代表python中对应的函数，&np_multiply_Cublas是对应的C++函数指针，之后的字符串是python中的函数doc
}



////////////////////////////////////////////////////////////////////////////////
