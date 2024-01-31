#include <optional>
#include  <stdexcept>

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

#include "pe/column_major/encoding_pipelined.h"
#include "index/expand_index_column_major.h"
#include "pe/column_major/pe_column_major_half2half.h"
#include "gemm/fp16.h"

# define blockDIM 256
# define debug 0

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)


__global__ void expand_index_encoding_pipelined_2half_2x_freq2(
        const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t outdata_height,
        const uint32_t indata_width,
        const uint32_t indata_valid_width,
        const int num_threads_col,
        const int in_offset,
        const int n_frequencies,
        half* indata_ptr,
        int* index,
        half2* outdata_expand, // pointer to rewrite data
        half* outdata_encoding // pointer to write encoding data
        )
{
        const uint64_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;
        /*
        相邻的线程处理同一列
        */
        const uint32_t i =  encoded_index % num_threads_col;
        const uint32_t j = encoded_index / num_threads_col;

        // Get the correspoding input data from index and sampled input
        half2 indata = expand_index_half2half_2x(i,  j,
                indata_width,
                in_offset,
                indata_ptr,
                index
                );
        // Rewrite the data into output
        outdata_expand[IDX2C(i, j, num_threads_col)] = indata;
        // Encoding and write
        frequency_encoding_device_2x_freq2(i, j,
        indata, n_frequencies,
        indata_valid_width,
        outdata_height,
        outdata_encoding);
}

void app_feature_expand_encoding_gemm_fp16_user_defined_multi_stream(
    torch::Tensor plane_mul_line_T, torch::Tensor basis_mat_weight, torch::Tensor fea_tmp,
    const int gemm_app_n, const int gemm_app_k, std::string gemm_app_activation,
    const at::optional<torch::Tensor>& app_latent, const std::optional<int> app_offset,
     torch::Tensor viewdirs, const int n_freque_fea, const int n_freque_view,
    const int fea_encoding_offset, const int view_encoding_offset, const int fea_data_offset, const int view_data_offset,
    const int padding_offset, const int padding_width,
    const int view_in_offset, const int view_valid_column, torch::Tensor index,
    const int tmp_height,  bool tmp_in_rowmajor, bool tmp_layer1out_rowmajor, bool tmp_layer2out_rowmajor, bool output_rowmajor,
    const int in_height, const int n1, const int k1, std::string activation1,
    const int n2, const int k2, std::string activation2,
    const int n3, const int k3, std::string activation3,
    torch::Tensor weight1,  torch::Tensor weight2, torch::Tensor weight3,
    torch::Tensor tmp_mlp_in, torch::Tensor tmp_layer1out,
    torch::Tensor tmp_layer2out, torch::Tensor output
){
    const int in_width_fea = gemm_app_n;
    const int in_width_view = viewdirs.size(1);
    const int tmp_width = k1;
    // Prepare for app_feature gemm
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm_app = select_gemm(true, false, gemm_app_activation);
    cutlass::half_t* plane_ptr = (cutlass::half_t*)plane_mul_line_T.data_ptr();
    cutlass::half_t* basis_mat_weight_ptr = (cutlass::half_t*)basis_mat_weight.data_ptr();
    cutlass::half_t* fea_tmp_ptr = (cutlass::half_t*)fea_tmp.data_ptr();
    // Prepare for app_latent copy
    int app_width;
    int num_elements_appCopy; int gridDIM_appCopy;
    half* app_ptr;

    // Prepare parameters for expand index and encoding
    int num_threads_col = tmp_height/2;
    int num_elements_fea = (in_width_fea)*num_threads_col;
    int num_elements_view = (view_valid_column)*num_threads_col;
    int gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    int gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;

    // Prepare data for gemm
    cutlass::half_t* output_begin = (cutlass::half_t*)(output.data_ptr());
    half* tmp_layer1out_ptr = reinterpret_cast<half*> (tmp_layer1out.data_ptr<torch::Half>());
    half* tmp_layer2out_ptr = reinterpret_cast<half*> (tmp_layer2out.data_ptr<torch::Half>());

    cutlass::half_t* in_ptr = (cutlass::half_t*)tmp_mlp_in.data_ptr();
    cutlass::half_t* tmp_layer1out_cutlassptr = (cutlass::half_t*)tmp_layer1out.data_ptr();
    cutlass::half_t* tmp_layer2out_cutlassptr = (cutlass::half_t*)tmp_layer2out.data_ptr();
    cutlass::half_t* weight1_ptr = (cutlass::half_t*)weight1.data_ptr();
    cutlass::half_t* weight2_ptr = (cutlass::half_t*)weight2.data_ptr();
    cutlass::half_t* weight3_ptr = (cutlass::half_t*)weight3.data_ptr();

    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm1 = select_gemm(tmp_in_rowmajor, tmp_layer1out_rowmajor, activation1);
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm2 = select_gemm(tmp_layer1out_rowmajor, tmp_layer2out_rowmajor, activation2);
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm3 = select_gemm(tmp_layer2out_rowmajor, output_rowmajor, activation3);

    const int iterations = (in_height + tmp_height -1 )/tmp_height;

    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    if (app_latent){
        app_width = app_latent.value().size(1);
        app_ptr = reinterpret_cast<half*> (app_latent.value().to(at::kHalf).data_ptr<torch::Half>());
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM,  0, stream[1]>>>(
                num_threads_col/4,
                num_elements_appCopy, //所有执行的 threads的个数 行*列
                app_width,  tmp_height,
                app_ptr,
                reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*app_offset.value()));
    }


    int final_height = in_height % tmp_height;
    int row_offset;
    for (int i = 0; i < iterations-1; i++)
    {
        row_offset = i*tmp_height;
        checkKernelErrors((gemm_app(plane_ptr + gemm_app_k*row_offset, basis_mat_weight_ptr, fea_tmp_ptr, tmp_height, gemm_app_n, gemm_app_k, stream[0])));

        // encoding for feature
        checkKernelErrors((encoding_pipelined_column_major_movedata_freq2_2x2x<<<gridDIM_fea, blockDIM, 0, stream[0]>>>
            (
            num_elements_fea, //所有执行的 threads的个数 行*列
            n_freque_fea,
            fea_encoding_offset, // 30 for features
            fea_data_offset,
            in_width_fea,
            tmp_width,
            tmp_height,
            num_threads_col, // 处理一列数据所需线程
            reinterpret_cast<half2*> (fea_tmp_ptr),
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
            )
        ));
        // expand index and encoding for viewdirs
        checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM, 0, stream[1]>>>
        (num_elements_view, tmp_height, in_width_view, view_valid_column, num_threads_col,
            view_in_offset,  n_freque_view,
            reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
            index.data_ptr<int>() + row_offset,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*tmp_height),
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*tmp_height)
        )));

        const int num_padding = num_threads_col*padding_width;
        const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;
        padding_to_zero_2x<<<gridDIM_padding, blockDIM,  0, stream[1]>>>(
            num_padding,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*padding_offset)
        );

        // cudaDeviceSynchronize();
        checkKernelErrors((gemm1(in_ptr , weight1_ptr,  tmp_layer1out_cutlassptr, tmp_height, n1, k1, std::nullopt)));
        checkKernelErrors((gemm2(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, tmp_height, n2, k2, std::nullopt)));
        checkKernelErrors((gemm3(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, tmp_height, n3, k3, std::nullopt)));
    }
    /*
    For the last stage, if the corresponding row is a multiple of 8, we calculate it directly, otherwise, we pad upwards to 8 before doing the calculation.
    */
    row_offset = (final_height % 8) == 0? (iterations-1)*tmp_height : (iterations-1)*tmp_height-(8-(final_height % 8));
    int finalstage_height = in_height - row_offset;

    num_threads_col = finalstage_height/2;
    num_elements_fea = (in_width_fea)*num_threads_col;
    num_elements_view = (view_valid_column)*num_threads_col;
    gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;
    const int num_padding = num_threads_col*padding_width;
    const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;

    // Gemm to get app_feature
    checkKernelErrors((gemm_app(plane_ptr + gemm_app_k*row_offset, basis_mat_weight_ptr, fea_tmp_ptr, finalstage_height, gemm_app_n, gemm_app_k, stream[0])));

    // encoding for feature
    checkKernelErrors((encoding_pipelined_column_major_movedata_freq2_2x2x<<<gridDIM_fea, blockDIM, 0, stream[0]>>>
    (
        num_elements_fea, //所有执行的 threads的个数 行*列
        n_freque_fea,
        fea_encoding_offset, // 30 for features
        fea_data_offset,
        in_width_fea,
        tmp_width,
        finalstage_height,
        num_threads_col, // 处理一列数据所需线程
        reinterpret_cast<half2*> (fea_tmp_ptr),
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
    )));
    // expand index and encoding for viewdirs
    checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM, 0, stream[1]>>>
    (num_elements_view, finalstage_height, in_width_view, view_valid_column, num_threads_col,
        view_in_offset,  n_freque_view,
        reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
        index.data_ptr<int>() + row_offset,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*finalstage_height),
        reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*finalstage_height)
    )));

    padding_to_zero_2x<<<gridDIM_padding, blockDIM,  0, stream[1]>>>(
        num_padding,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*padding_offset)
    );

    if (app_latent){
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM,  0, stream[1]>>>(
            num_threads_col/4,
            num_elements_appCopy, //所有执行的 threads的个数 行*列
            app_width, finalstage_height,
            app_ptr ,
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*app_offset.value())
        );
    }
    // cudaDeviceSynchronize();
    checkKernelErrors((gemm1(in_ptr, weight1_ptr, tmp_layer1out_cutlassptr, finalstage_height, n1, k1, std::nullopt)));
    checkKernelErrors((gemm2(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, finalstage_height, n2, k2, std::nullopt)));
    checkKernelErrors((gemm3(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, finalstage_height, n3, k3, std::nullopt)));
}



void app_feature_expand_encoding_gemm_fp16_user_defined_1stream(
    torch::Tensor plane_mul_line_T, torch::Tensor basis_mat_weight, torch::Tensor fea_tmp,
    const int gemm_app_n, const int gemm_app_k, std::string gemm_app_activation,
    const at::optional<torch::Tensor>& app_latent, const std::optional<int> app_offset,
     torch::Tensor viewdirs, const int n_freque_fea, const int n_freque_view,
    const int fea_encoding_offset, const int view_encoding_offset, const int fea_data_offset, const int view_data_offset,
    const int padding_offset, const int padding_width,
    const int view_in_offset, const int view_valid_column, torch::Tensor index,
    const int tmp_height,  bool tmp_in_rowmajor, bool tmp_layer1out_rowmajor, bool tmp_layer2out_rowmajor, bool output_rowmajor,
    const int in_height, const int n1, const int k1, std::string activation1,
    const int n2, const int k2, std::string activation2,
    const int n3, const int k3, std::string activation3,
    torch::Tensor weight1,  torch::Tensor weight2, torch::Tensor weight3,
    torch::Tensor tmp_mlp_in, torch::Tensor tmp_layer1out,
    torch::Tensor tmp_layer2out, torch::Tensor output
){
    const int in_width_fea = gemm_app_n;
    const int in_width_view = viewdirs.size(1);
    const int tmp_width = k1;
    // Prepare for app_feature gemm
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm_app = select_gemm(true, false, gemm_app_activation);
    cutlass::half_t* plane_ptr = (cutlass::half_t*)plane_mul_line_T.data_ptr();
    cutlass::half_t* basis_mat_weight_ptr = (cutlass::half_t*)basis_mat_weight.data_ptr();
    cutlass::half_t* fea_tmp_ptr = (cutlass::half_t*)fea_tmp.data_ptr();
    // Prepare for app_latent copy
    int app_width;
    int num_elements_appCopy; int gridDIM_appCopy;
    half* app_ptr;

    // Prepare parameters for expand index and encoding
    int num_threads_col = tmp_height/2;
    int num_elements_fea = (in_width_fea)*num_threads_col;
    int num_elements_view = (view_valid_column)*num_threads_col;
    int gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    int gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;

    // Prepare data for gemm
    cutlass::half_t* output_begin = (cutlass::half_t*)(output.data_ptr());
    half* tmp_layer1out_ptr = reinterpret_cast<half*> (tmp_layer1out.data_ptr<torch::Half>());
    half* tmp_layer2out_ptr = reinterpret_cast<half*> (tmp_layer2out.data_ptr<torch::Half>());

    cutlass::half_t* in_ptr = (cutlass::half_t*)tmp_mlp_in.data_ptr();
    cutlass::half_t* tmp_layer1out_cutlassptr = (cutlass::half_t*)tmp_layer1out.data_ptr();
    cutlass::half_t* tmp_layer2out_cutlassptr = (cutlass::half_t*)tmp_layer2out.data_ptr();
    cutlass::half_t* weight1_ptr = (cutlass::half_t*)weight1.data_ptr();
    cutlass::half_t* weight2_ptr = (cutlass::half_t*)weight2.data_ptr();
    cutlass::half_t* weight3_ptr = (cutlass::half_t*)weight3.data_ptr();

    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm1 = select_gemm(tmp_in_rowmajor, tmp_layer1out_rowmajor, activation1);
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm2 = select_gemm(tmp_layer1out_rowmajor, tmp_layer2out_rowmajor, activation2);
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm3 = select_gemm(tmp_layer2out_rowmajor, output_rowmajor, activation3);

    const int iterations = (in_height + tmp_height -1 )/tmp_height;


    if (app_latent){
        app_width = app_latent.value().size(1);
        app_ptr = reinterpret_cast<half*> (app_latent.value().to(at::kHalf).data_ptr<torch::Half>());
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM>>>(
                num_threads_col/4,
                num_elements_appCopy, //所有执行的 threads的个数 行*列
                app_width,  tmp_height,
                app_ptr,
                reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*app_offset.value()));
    }


    int final_height = in_height % tmp_height;
    int row_offset;
    for (int i = 0; i < iterations-1; i++)
    {
        row_offset = i*tmp_height;
        checkKernelErrors((gemm_app(plane_ptr + gemm_app_k*row_offset, basis_mat_weight_ptr, fea_tmp_ptr, tmp_height, gemm_app_n, gemm_app_k, std::nullopt)));

        // encoding for feature
        checkKernelErrors((encoding_pipelined_column_major_movedata_freq2_2x2x<<<gridDIM_fea, blockDIM>>>
            (
            num_elements_fea, //所有执行的 threads的个数 行*列
            n_freque_fea,
            fea_encoding_offset, // 30 for features
            fea_data_offset,
            in_width_fea,
            tmp_width,
            tmp_height,
            num_threads_col, // 处理一列数据所需线程
            reinterpret_cast<half2*> (fea_tmp_ptr),
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
            )
        ));
        // expand index and encoding for viewdirs
        checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM>>>
        (num_elements_view, tmp_height, in_width_view, view_valid_column, num_threads_col,
            view_in_offset,  n_freque_view,
            reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
            index.data_ptr<int>() + row_offset,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*tmp_height),
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*tmp_height)
        )));

        const int num_padding = num_threads_col*padding_width;
        const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;
        checkKernelErrors((padding_to_zero_2x<<<gridDIM_padding, blockDIM>>>(
            num_padding,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*padding_offset)
        )));

        // cudaDeviceSynchronize();
        checkKernelErrors((gemm1(in_ptr , weight1_ptr,  tmp_layer1out_cutlassptr, tmp_height, n1, k1, std::nullopt)));
        checkKernelErrors((gemm2(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, tmp_height, n2, k2, std::nullopt)));
        checkKernelErrors((gemm3(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, tmp_height, n3, k3, std::nullopt)));
    }
    /*
    For the last stage, if the corresponding row is a multiple of 8, we calculate it directly, otherwise, we pad upwards to 8 before doing the calculation.
    */
    row_offset = (final_height % 8) == 0? (iterations-1)*tmp_height : (iterations-1)*tmp_height-(8-(final_height % 8));
    int finalstage_height = in_height - row_offset;

    num_threads_col = finalstage_height/2;
    num_elements_fea = (in_width_fea)*num_threads_col;
    num_elements_view = (view_valid_column)*num_threads_col;
    gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;
    const int num_padding = num_threads_col*padding_width;
    const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;

    // Gemm to get app_feature
    checkKernelErrors((gemm_app(plane_ptr + gemm_app_k*row_offset, basis_mat_weight_ptr, fea_tmp_ptr, finalstage_height, gemm_app_n, gemm_app_k, std::nullopt)));

    // encoding for feature
    checkKernelErrors((encoding_pipelined_column_major_movedata_freq2_2x2x<<<gridDIM_fea, blockDIM>>>
    (
        num_elements_fea, //所有执行的 threads的个数 行*列
        n_freque_fea,
        fea_encoding_offset, // 30 for features
        fea_data_offset,
        in_width_fea,
        tmp_width,
        finalstage_height,
        num_threads_col, // 处理一列数据所需线程
        reinterpret_cast<half2*> (fea_tmp_ptr),
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
    )));
    // expand index and encoding for viewdirs
    checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM>>>
    (num_elements_view, finalstage_height, in_width_view, view_valid_column, num_threads_col,
        view_in_offset,  n_freque_view,
        reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
        index.data_ptr<int>() + row_offset,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*finalstage_height),
        reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*finalstage_height)
    )));

    padding_to_zero_2x<<<gridDIM_padding, blockDIM>>>(
        num_padding,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*padding_offset)
    );

    if (app_latent){
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM>>>(
            num_threads_col/4,
            num_elements_appCopy, //所有执行的 threads的个数 行*列
            app_width, finalstage_height,
            app_ptr ,
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*app_offset.value())
        );
    }
    // cudaDeviceSynchronize();
    checkKernelErrors((gemm1(in_ptr, weight1_ptr, tmp_layer1out_cutlassptr, finalstage_height, n1, k1, std::nullopt)));
    checkKernelErrors((gemm2(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, finalstage_height, n2, k2, std::nullopt)));
    checkKernelErrors((gemm3(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, finalstage_height, n3, k3, std::nullopt)));
}

void app_feature_expand_encoding_gemm_fp16(
    torch::Tensor plane_mul_line_T, torch::Tensor basis_mat_weight, torch::Tensor fea_tmp,
    const int gemm_app_n, const int gemm_app_k, std::string gemm_app_activation,
    const at::optional<torch::Tensor>& app_latent, const std::optional<int> app_offset,
     torch::Tensor viewdirs, const int n_freque_fea, const int n_freque_view,
    const int fea_encoding_offset, const int view_encoding_offset, const int fea_data_offset, const int view_data_offset,
    const int padding_offset, const int padding_width,
    const int view_in_offset, const int view_valid_column, torch::Tensor index,
    const int tmp_height,  bool tmp_in_rowmajor, bool tmp_layer1out_rowmajor, bool tmp_layer2out_rowmajor, bool output_rowmajor,
    const int in_height, const int n1, const int k1, std::string activation1,
    const int n2, const int k2, std::string activation2,
    const int n3, const int k3, std::string activation3,
    torch::Tensor weight1,  torch::Tensor weight2, torch::Tensor weight3,
    torch::Tensor tmp_mlp_in, torch::Tensor tmp_layer1out,
    torch::Tensor tmp_layer2out, torch::Tensor output
){
    const int in_width_fea = gemm_app_n;
    const int in_width_view = viewdirs.size(1);
    const int tmp_width = k1;

    cutlass::half_t* plane_ptr = (cutlass::half_t*)plane_mul_line_T.data_ptr();
    cutlass::half_t* basis_mat_weight_ptr = (cutlass::half_t*)basis_mat_weight.data_ptr();
    cutlass::half_t* fea_tmp_ptr = (cutlass::half_t*)fea_tmp.data_ptr();
    // Prepare for app_latent copy
    int app_width;
    int num_elements_appCopy; int gridDIM_appCopy;
    half* app_ptr;

    // Prepare parameters for expand index and encoding
    int num_threads_col = tmp_height/2;
    int num_elements_fea = (in_width_fea)*num_threads_col;
    int num_elements_view = (view_valid_column)*num_threads_col;
    int gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    int gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;

    // Prepare data for gemm
    cutlass::half_t* output_begin = (cutlass::half_t*)(output.data_ptr());
    half* tmp_layer1out_ptr = reinterpret_cast<half*> (tmp_layer1out.data_ptr<torch::Half>());
    half* tmp_layer2out_ptr = reinterpret_cast<half*> (tmp_layer2out.data_ptr<torch::Half>());

    cutlass::half_t* in_ptr = (cutlass::half_t*)tmp_mlp_in.data_ptr();
    cutlass::half_t* tmp_layer1out_cutlassptr = (cutlass::half_t*)tmp_layer1out.data_ptr();
    cutlass::half_t* tmp_layer2out_cutlassptr = (cutlass::half_t*)tmp_layer2out.data_ptr();
    cutlass::half_t* weight1_ptr = (cutlass::half_t*)weight1.data_ptr();
    cutlass::half_t* weight2_ptr = (cutlass::half_t*)weight2.data_ptr();
    cutlass::half_t* weight3_ptr = (cutlass::half_t*)weight3.data_ptr();



    const int iterations = (in_height + tmp_height -1 )/tmp_height;


    if (app_latent){
        app_width = app_latent.value().size(1);
        app_ptr = reinterpret_cast<half*> (app_latent.value().to(at::kHalf).data_ptr<torch::Half>());
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM>>>(
                num_threads_col/4,
                num_elements_appCopy, //所有执行的 threads的个数 行*列
                app_width,  tmp_height,
                app_ptr,
                reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*app_offset.value()));
    }


    int final_height = in_height % tmp_height;
    int row_offset;
    for (int i = 0; i < iterations-1; i++)
    {
        row_offset = i*tmp_height;
        checkKernelErrors((gemm_fp16_row_col_col_4x_problemsize<EpilogueOp, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(plane_ptr + gemm_app_k*row_offset, basis_mat_weight_ptr, fea_tmp_ptr, tmp_height, gemm_app_n, gemm_app_k, std::nullopt)));

        // encoding for feature
        checkKernelErrors((encoding_pipelined_column_major_movedata_freq2_2x2x<<<gridDIM_fea, blockDIM>>>
            (
            num_elements_fea, //所有执行的 threads的个数 行*列
            n_freque_fea,
            fea_encoding_offset, // 30 for features
            fea_data_offset,
            in_width_fea,
            tmp_width,
            tmp_height,
            num_threads_col, // 处理一列数据所需线程
            reinterpret_cast<half2*> (fea_tmp_ptr),
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
            )
        ));
        // expand index and encoding for viewdirs
        checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM>>>
        (num_elements_view, tmp_height, in_width_view, view_valid_column, num_threads_col,
            view_in_offset,  n_freque_view,
            reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
            index.data_ptr<int>() + row_offset,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*tmp_height),
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*tmp_height)
        )));

        const int num_padding = num_threads_col*padding_width;
        const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;
        checkKernelErrors((padding_to_zero_2x<<<gridDIM_padding, blockDIM>>>(
            num_padding,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*padding_offset)
        )));

        // cudaDeviceSynchronize();
        checkKernelErrors((gemm_fp16_col_col_col_4x_problemsize<EpilogueOpRelu, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(in_ptr , weight1_ptr,  tmp_layer1out_cutlassptr, tmp_height, n1, k1, std::nullopt)));
        checkKernelErrors((gemm_fp16_col_col_col_4x_problemsize<EpilogueOpRelu, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, tmp_height, n2, k2, std::nullopt)));
        checkKernelErrors((gemm_fp16_col_col_row_4x_problemsize<EpilogueOpSigmoid, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, tmp_height, n3, k3, std::nullopt)));
    }
    /*
    For the last stage, if the corresponding row is a multiple of 8, we calculate it directly, otherwise, we pad upwards to 8 before doing the calculation.
    */
    row_offset = (final_height % 8) == 0? (iterations-1)*tmp_height : (iterations-1)*tmp_height-(8-(final_height % 8));
    int finalstage_height = in_height - row_offset;

    num_threads_col = finalstage_height/2;
    num_elements_fea = (in_width_fea)*num_threads_col;
    num_elements_view = (view_valid_column)*num_threads_col;
    gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;
    const int num_padding = num_threads_col*padding_width;
    const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;

    // Gemm to get app_feature
    checkKernelErrors((gemm_fp16_row_col_col_4x_problemsize<EpilogueOp, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(plane_ptr + gemm_app_k*row_offset, basis_mat_weight_ptr, fea_tmp_ptr, finalstage_height, gemm_app_n, gemm_app_k, std::nullopt)));

    // encoding for feature
    checkKernelErrors((encoding_pipelined_column_major_movedata_freq2_2x2x<<<gridDIM_fea, blockDIM>>>
    (
        num_elements_fea, //所有执行的 threads的个数 行*列
        n_freque_fea,
        fea_encoding_offset, // 30 for features
        fea_data_offset,
        in_width_fea,
        tmp_width,
        finalstage_height,
        num_threads_col, // 处理一列数据所需线程
        reinterpret_cast<half2*> (fea_tmp_ptr),
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>())
    )));
    // expand index and encoding for viewdirs
    checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM>>>
    (num_elements_view, finalstage_height, in_width_view, view_valid_column, num_threads_col,
        view_in_offset,  n_freque_view,
        reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
        index.data_ptr<int>() + row_offset,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*finalstage_height),
        reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*finalstage_height)
    )));

    padding_to_zero_2x<<<gridDIM_padding, blockDIM>>>(
        num_padding,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*padding_offset)
    );

    if (app_latent){
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM>>>(
            num_threads_col/4,
            num_elements_appCopy, //所有执行的 threads的个数 行*列
            app_width, finalstage_height,
            app_ptr ,
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*app_offset.value())
        );
    }
    // cudaDeviceSynchronize();
    checkKernelErrors((gemm_fp16_col_col_col_4x_problemsize<EpilogueOpRelu, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(in_ptr, weight1_ptr, tmp_layer1out_cutlassptr, finalstage_height, n1, k1, std::nullopt)));
    checkKernelErrors((gemm_fp16_col_col_col_4x_problemsize<EpilogueOpRelu, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, finalstage_height, n2, k2, std::nullopt)));
    checkKernelErrors((gemm_fp16_col_col_row_4x_problemsize<EpilogueOpSigmoid, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, finalstage_height, n3, k3, std::nullopt)));
}


void expand_encoding_gemm_fp16_user_defined_multi_stream(
    const at::optional<torch::Tensor>& app_latent, const std::optional<int> app_offset,
    torch::Tensor features, torch::Tensor viewdirs, const int n_freque_fea, const int n_freque_view,
    const int fea_encoding_offset, const int view_encoding_offset, const int fea_data_offset, const int view_data_offset,
    const int padding_offset, const int padding_width,
    const int view_in_offset, const int view_valid_column, torch::Tensor index,
    const int tmp_height,  bool tmp_in_rowmajor, bool tmp_layer1out_rowmajor, bool tmp_layer2out_rowmajor, bool output_rowmajor,
    const int in_height, const int n1, const int k1, std::string activation1,
    const int n2, const int k2, std::string activation2,
    const int n3, const int k3, std::string activation3,
    torch::Tensor weight1,  torch::Tensor weight2, torch::Tensor weight3,
    torch::Tensor tmp_mlp_in, torch::Tensor tmp_layer1out,
    torch::Tensor tmp_layer2out, torch::Tensor output
){
    const int in_width_fea = features.size(1);
    const int in_width_view = viewdirs.size(1);
    const int tmp_width = k1;
    // Prepare for app_latent copy
    int app_width;
    int num_elements_appCopy; int gridDIM_appCopy;
    half* app_ptr;

    // Prepare parameters for expand index and encoding
    int num_threads_col = tmp_height/2;
    int num_elements_fea = (in_width_fea)*num_threads_col;
    int num_elements_view = (view_valid_column)*num_threads_col;
    int gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    int gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;

    // Prepare data for gemm
    cutlass::half_t* output_begin = (cutlass::half_t*)(output.data_ptr());
    half* tmp_layer1out_ptr = reinterpret_cast<half*> (tmp_layer1out.data_ptr<torch::Half>());
    half* tmp_layer2out_ptr = reinterpret_cast<half*> (tmp_layer2out.data_ptr<torch::Half>());

    cutlass::half_t* in_ptr = (cutlass::half_t*)tmp_mlp_in.data_ptr();
    cutlass::half_t* tmp_layer1out_cutlassptr = (cutlass::half_t*)tmp_layer1out.data_ptr();
    cutlass::half_t* tmp_layer2out_cutlassptr = (cutlass::half_t*)tmp_layer2out.data_ptr();
    cutlass::half_t* weight1_ptr = (cutlass::half_t*)weight1.data_ptr();
    cutlass::half_t* weight2_ptr = (cutlass::half_t*)weight2.data_ptr();
    cutlass::half_t* weight3_ptr = (cutlass::half_t*)weight3.data_ptr();

    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm1 = select_gemm(tmp_in_rowmajor, tmp_layer1out_rowmajor, activation1);
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm2 = select_gemm(tmp_layer1out_rowmajor, tmp_layer2out_rowmajor, activation2);
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm3 = select_gemm(tmp_layer2out_rowmajor, output_rowmajor, activation3);

    const int iterations = (in_height + tmp_height -1 )/tmp_height;

    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    if (app_latent){
        app_width = app_latent.value().size(1);
        app_ptr = reinterpret_cast<half*> (app_latent.value().to(at::kHalf).data_ptr<torch::Half>());
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM,  0, stream[1]>>>(
                num_threads_col/4,
                num_elements_appCopy, //所有执行的 threads的个数 行*列
                app_width,  tmp_height,
                app_ptr,
                reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*app_offset.value()));
    }


    int final_height = in_height % tmp_height;
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
        // expand index and encoding for viewdirs
        checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM, 0, stream[1]>>>
        (num_elements_view, tmp_height, in_width_view, view_valid_column, num_threads_col,
            view_in_offset,  n_freque_view,
            reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
            index.data_ptr<int>() + row_offset,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*tmp_height),
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*tmp_height)
        )));

        const int num_padding = num_threads_col*padding_width;
        const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;
        padding_to_zero_2x<<<gridDIM_padding, blockDIM,  0, stream[1]>>>(
            num_padding,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*padding_offset)
        );

        // cudaDeviceSynchronize();
        checkKernelErrors((gemm1(in_ptr , weight1_ptr,  tmp_layer1out_cutlassptr, tmp_height, n1, k1, std::nullopt)));
        checkKernelErrors((gemm2(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, tmp_height, n2, k2, std::nullopt)));
        checkKernelErrors((gemm3(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, tmp_height, n3, k3, std::nullopt)));
    }
    /*
    For the last stage, if the corresponding row is a multiple of 8, we calculate it directly, otherwise, we pad upwards to 8 before doing the calculation.
    */
    row_offset = (final_height % 8) == 0? (iterations-1)*tmp_height : (iterations-1)*tmp_height-(8-(final_height % 8));
    int finalstage_height = in_height - row_offset;

    num_threads_col = finalstage_height/2;
    num_elements_fea = (in_width_fea)*num_threads_col;
    num_elements_view = (view_valid_column)*num_threads_col;
    gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;
    const int num_padding = num_threads_col*padding_width;
    const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;

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
    // expand index and encoding for viewdirs
    checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM, 0, stream[1]>>>
    (num_elements_view, finalstage_height, in_width_view, view_valid_column, num_threads_col,
        view_in_offset,  n_freque_view,
        reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
        index.data_ptr<int>() + row_offset,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*finalstage_height),
        reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*finalstage_height)
    )));

    padding_to_zero_2x<<<gridDIM_padding, blockDIM,  0, stream[1]>>>(
        num_padding,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*padding_offset)
    );

    if (app_latent){
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM,  0, stream[1]>>>(
            num_threads_col/4,
            num_elements_appCopy, //所有执行的 threads的个数 行*列
            app_width, finalstage_height,
            app_ptr ,
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*app_offset.value())
        );
    }
    // cudaDeviceSynchronize();
    checkKernelErrors((gemm1(in_ptr, weight1_ptr, tmp_layer1out_cutlassptr, finalstage_height, n1, k1, std::nullopt)));
    checkKernelErrors((gemm2(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, finalstage_height, n2, k2, std::nullopt)));
    checkKernelErrors((gemm3(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, finalstage_height, n3, k3, std::nullopt)));
}

void expand_encoding_gemm_fp16_user_defined_1stream(
    const at::optional<torch::Tensor>& app_latent, const std::optional<int> app_offset,
    torch::Tensor features, torch::Tensor viewdirs, const int n_freque_fea, const int n_freque_view,
    const int fea_encoding_offset, const int view_encoding_offset, const int fea_data_offset, const int view_data_offset,
    const int padding_offset, const int padding_width,
    const int view_in_offset, const int view_valid_column, torch::Tensor index,
    const int tmp_height,  bool tmp_in_rowmajor, bool tmp_layer1out_rowmajor, bool tmp_layer2out_rowmajor, bool output_rowmajor,
    const int in_height, const int n1, const int k1, std::string activation1,
    const int n2, const int k2, std::string activation2,
    const int n3, const int k3, std::string activation3,
    torch::Tensor weight1,  torch::Tensor weight2, torch::Tensor weight3,
    torch::Tensor tmp_mlp_in, torch::Tensor tmp_layer1out,
    torch::Tensor tmp_layer2out, torch::Tensor output
){
    const int in_width_fea = features.size(1);
    const int in_width_view = viewdirs.size(1);
    const int tmp_width = k1;
    // Prepare for app_latent copy
    int app_width;
    int num_elements_appCopy; int gridDIM_appCopy;
    half* app_ptr;

    // Prepare parameters for expand index and encoding
    int num_threads_col = tmp_height/2;
    int num_elements_fea = (in_width_fea)*num_threads_col;
    int num_elements_view = (view_valid_column)*num_threads_col;
    int gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    int gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;

    // Prepare data for gemm
    cutlass::half_t* output_begin = (cutlass::half_t*)(output.data_ptr());
    half* tmp_layer1out_ptr = reinterpret_cast<half*> (tmp_layer1out.data_ptr<torch::Half>());
    half* tmp_layer2out_ptr = reinterpret_cast<half*> (tmp_layer2out.data_ptr<torch::Half>());

    cutlass::half_t* in_ptr = (cutlass::half_t*)tmp_mlp_in.data_ptr();
    cutlass::half_t* tmp_layer1out_cutlassptr = (cutlass::half_t*)tmp_layer1out.data_ptr();
    cutlass::half_t* tmp_layer2out_cutlassptr = (cutlass::half_t*)tmp_layer2out.data_ptr();
    cutlass::half_t* weight1_ptr = (cutlass::half_t*)weight1.data_ptr();
    cutlass::half_t* weight2_ptr = (cutlass::half_t*)weight2.data_ptr();
    cutlass::half_t* weight3_ptr = (cutlass::half_t*)weight3.data_ptr();

    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm1 = select_gemm(tmp_in_rowmajor, tmp_layer1out_rowmajor, activation1);
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm2 = select_gemm(tmp_layer1out_rowmajor, tmp_layer2out_rowmajor, activation2);
    std::function<void(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*, int, int, int, const std::optional<cudaStream_t>)> gemm3 = select_gemm(tmp_layer2out_rowmajor, output_rowmajor, activation3);

    const int iterations = (in_height + tmp_height -1 )/tmp_height;



    if (app_latent){
        app_width = app_latent.value().size(1);
        app_ptr = reinterpret_cast<half*> (app_latent.value().to(at::kHalf).data_ptr<torch::Half>());
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM>>>(
                num_threads_col/4,
                num_elements_appCopy, //所有执行的 threads的个数 行*列
                app_width,  tmp_height,
                app_ptr,
                reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*app_offset.value()));
    }


    int final_height = in_height % tmp_height;
    int row_offset;
    for (int i = 0; i < iterations-1; i++)
    {
        row_offset = i*tmp_height;
        // encoding for feature
        encoding_pipelined_column_major_movedata_freq2_2x<<<gridDIM_fea, blockDIM>>>
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
        // expand index and encoding for viewdirs
        checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM>>>
        (num_elements_view, tmp_height, in_width_view, view_valid_column, num_threads_col,
            view_in_offset,  n_freque_view,
            reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
            index.data_ptr<int>() + row_offset,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*tmp_height),
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*tmp_height)
        )));

        const int num_padding = num_threads_col*padding_width;
        const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;
        padding_to_zero_2x<<<gridDIM_padding, blockDIM>>>(
            num_padding,
            reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + tmp_height*padding_offset)
        );

        // cudaDeviceSynchronize();
        checkKernelErrors((gemm1(in_ptr , weight1_ptr,  tmp_layer1out_cutlassptr, tmp_height, n1, k1, std::nullopt)));
        checkKernelErrors((gemm2(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, tmp_height, n2, k2, std::nullopt)));
        checkKernelErrors((gemm3(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, tmp_height, n3, k3, std::nullopt)));
    }
    /*
    For the last stage, if the corresponding row is a multiple of 8, we calculate it directly, otherwise, we pad upwards to 8 before doing the calculation.
    */
    row_offset = (final_height % 8) == 0? (iterations-1)*tmp_height : (iterations-1)*tmp_height-(8-(final_height % 8));
    int finalstage_height = in_height - row_offset;

    num_threads_col = finalstage_height/2;
    num_elements_fea = (in_width_fea)*num_threads_col;
    num_elements_view = (view_valid_column)*num_threads_col;
    gridDIM_fea = (num_elements_fea + blockDIM-1)/blockDIM;
    gridDIM_view = (num_elements_view + blockDIM-1)/blockDIM;
    const int num_padding = num_threads_col*padding_width;
    const int gridDIM_padding = (num_padding + blockDIM-1)/blockDIM;

    // encoding for feature
    encoding_pipelined_column_major_movedata_freq2_2x<<<gridDIM_fea, blockDIM>>>
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
    // expand index and encoding for viewdirs
    checkKernelErrors((expand_index_encoding_pipelined_2half_2x_freq2<<<gridDIM_view, blockDIM>>>
    (num_elements_view, finalstage_height, in_width_view, view_valid_column, num_threads_col,
        view_in_offset,  n_freque_view,
        reinterpret_cast<half*> (viewdirs.to(at::kHalf).data_ptr<torch::Half>()),
        index.data_ptr<int>() + row_offset,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_data_offset*finalstage_height),
        reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>()+ view_encoding_offset*finalstage_height)
    )));

    padding_to_zero_2x<<<gridDIM_padding, blockDIM>>>(
        num_padding,
        reinterpret_cast<half2*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*padding_offset)
    );

    if (app_latent){
        num_elements_appCopy = app_width*num_threads_col/4;
        gridDIM_appCopy = (num_elements_appCopy + blockDIM-1)/blockDIM;
        copy_shared_column_8x<<<gridDIM_appCopy, blockDIM>>>(
            num_threads_col/4,
            num_elements_appCopy, //所有执行的 threads的个数 行*列
            app_width, finalstage_height,
            app_ptr ,
            reinterpret_cast<half*> (tmp_mlp_in.data_ptr<torch::Half>() + finalstage_height*app_offset.value())
        );
    }
    // cudaDeviceSynchronize();
    checkKernelErrors((gemm1(in_ptr, weight1_ptr, tmp_layer1out_cutlassptr, finalstage_height, n1, k1, std::nullopt)));
    checkKernelErrors((gemm2(tmp_layer1out_cutlassptr, weight2_ptr, tmp_layer2out_cutlassptr, finalstage_height, n2, k2, std::nullopt)));
    checkKernelErrors((gemm3(tmp_layer2out_cutlassptr, weight3_ptr,  output_begin + n3*row_offset, finalstage_height, n3, k3, std::nullopt)));
}

void  expand_encoding_gemm_fp16_user_defined(
    const at::optional<torch::Tensor>& app_latent, const std::optional<int> app_offset,
    torch::Tensor features, torch::Tensor viewdirs, const int n_freque_fea, const int n_freque_view,
    const int fea_encoding_offset, const int view_encoding_offset, const int fea_data_offset, const int view_data_offset,
    const int padding_offset, const int padding_width,
    const int view_in_offset, const int view_valid_column, torch::Tensor index,
    const int tmp_height,  bool tmp_in_rowmajor, bool tmp_layer1out_rowmajor, bool tmp_layer2out_rowmajor, bool output_rowmajor,
    const int in_height, const int n1, const int k1, std::string activation1,
    const int n2, const int k2, std::string activation2,
    const int n3, const int k3, std::string activation3,
    torch::Tensor weight1,  torch::Tensor weight2, torch::Tensor weight3,
    torch::Tensor tmp_mlp_in, torch::Tensor tmp_layer1out,
    torch::Tensor tmp_layer2out, torch::Tensor output, bool multiStream){
        if (multiStream){
            expand_encoding_gemm_fp16_user_defined_multi_stream(
                app_latent, app_offset,
                features, viewdirs, n_freque_fea, n_freque_view,
                fea_encoding_offset, view_encoding_offset, fea_data_offset, view_data_offset,
                padding_offset,  padding_width,
                view_in_offset, view_valid_column, index,
                tmp_height, tmp_in_rowmajor, tmp_layer1out_rowmajor, tmp_layer2out_rowmajor,output_rowmajor,
                in_height, n1,  k1, activation1,
                n2, k2, activation2,
                n3, k3, activation3,
                weight1,  weight2, weight3,
                tmp_mlp_in, tmp_layer1out,
                tmp_layer2out, output);
        }
        else{
            expand_encoding_gemm_fp16_user_defined_1stream(
                app_latent, app_offset,
                features, viewdirs, n_freque_fea, n_freque_view,
                fea_encoding_offset, view_encoding_offset, fea_data_offset, view_data_offset,
                padding_offset,  padding_width,
                view_in_offset, view_valid_column, index,
                tmp_height, tmp_in_rowmajor, tmp_layer1out_rowmajor, tmp_layer2out_rowmajor,output_rowmajor,
                in_height, n1,  k1, activation1,
                n2, k2, activation2,
                n3, k3, activation3,
                weight1,  weight2, weight3,
                tmp_mlp_in, tmp_layer1out,
                tmp_layer2out, output);
        }

}

void app_feature_expand_encoding_gemm_fp16_user_defined(
    torch::Tensor plane_mul_line_T, torch::Tensor basis_mat_weight, torch::Tensor fea_tmp,
    const int gemm_app_n, const int gemm_app_k, std::string gemm_app_activation,
    const at::optional<torch::Tensor>& app_latent, const std::optional<int> app_offset,
     torch::Tensor viewdirs, const int n_freque_fea, const int n_freque_view,
    const int fea_encoding_offset, const int view_encoding_offset, const int fea_data_offset, const int view_data_offset,
    const int padding_offset, const int padding_width,
    const int view_in_offset, const int view_valid_column, torch::Tensor index,
    const int tmp_height,  bool tmp_in_rowmajor, bool tmp_layer1out_rowmajor, bool tmp_layer2out_rowmajor, bool output_rowmajor,
    const int in_height, const int n1, const int k1, std::string activation1,
    const int n2, const int k2, std::string activation2,
    const int n3, const int k3, std::string activation3,
    torch::Tensor weight1,  torch::Tensor weight2, torch::Tensor weight3,
    torch::Tensor tmp_mlp_in, torch::Tensor tmp_layer1out,
    torch::Tensor tmp_layer2out, torch::Tensor output, bool multiStream){
        if (multiStream){
            app_feature_expand_encoding_gemm_fp16_user_defined_multi_stream(
            plane_mul_line_T, basis_mat_weight, fea_tmp,
            gemm_app_n, gemm_app_k, gemm_app_activation,
            app_latent, app_offset,
            viewdirs, n_freque_fea, n_freque_view,
            fea_encoding_offset, view_encoding_offset, fea_data_offset, view_data_offset,
            padding_offset,  padding_width,
            view_in_offset, view_valid_column, index,
            tmp_height, tmp_in_rowmajor, tmp_layer1out_rowmajor, tmp_layer2out_rowmajor,output_rowmajor,
            in_height, n1,  k1, activation1,
            n2, k2, activation2,
            n3, k3, activation3,
            weight1,  weight2, weight3,
            tmp_mlp_in, tmp_layer1out,
            tmp_layer2out, output);
        }
        else{
            app_feature_expand_encoding_gemm_fp16_user_defined_1stream(
            plane_mul_line_T, basis_mat_weight, fea_tmp,
            gemm_app_n, gemm_app_k, gemm_app_activation,
            app_latent, app_offset,
            viewdirs, n_freque_fea, n_freque_view,
            fea_encoding_offset, view_encoding_offset, fea_data_offset, view_data_offset,
            padding_offset,  padding_width,
            view_in_offset, view_valid_column, index,
            tmp_height, tmp_in_rowmajor, tmp_layer1out_rowmajor, tmp_layer2out_rowmajor,output_rowmajor,
            in_height, n1,  k1, activation1,
            n2, k2, activation2,
            n3, k3, activation3,
            weight1,  weight2, weight3,
            tmp_mlp_in, tmp_layer1out,
            tmp_layer2out, output);
        }

}

void app_feature_expand_encoding_gemm_fp16_cpp(
    torch::Tensor plane_mul_line_T, torch::Tensor basis_mat_weight, torch::Tensor fea_tmp,
    const int gemm_app_n, const int gemm_app_k, std::string gemm_app_activation,
    const at::optional<torch::Tensor>& app_latent, const std::optional<int> app_offset,
     torch::Tensor viewdirs, const int n_freque_fea, const int n_freque_view,
    const int fea_encoding_offset, const int view_encoding_offset, const int fea_data_offset, const int view_data_offset,
    const int padding_offset, const int padding_width,
    const int view_in_offset, const int view_valid_column, torch::Tensor index,
    const int tmp_height,  bool tmp_in_rowmajor, bool tmp_layer1out_rowmajor, bool tmp_layer2out_rowmajor, bool output_rowmajor,
    const int in_height, const int n1, const int k1, std::string activation1,
    const int n2, const int k2, std::string activation2,
    const int n3, const int k3, std::string activation3,
    torch::Tensor weight1,  torch::Tensor weight2, torch::Tensor weight3,
    torch::Tensor tmp_mlp_in, torch::Tensor tmp_layer1out,
    torch::Tensor tmp_layer2out, torch::Tensor output, bool multiStream){

            app_feature_expand_encoding_gemm_fp16(
            plane_mul_line_T, basis_mat_weight, fea_tmp,
            gemm_app_n, gemm_app_k, gemm_app_activation,
            app_latent, app_offset,
            viewdirs, n_freque_fea, n_freque_view,
            fea_encoding_offset, view_encoding_offset, fea_data_offset, view_data_offset,
            padding_offset,  padding_width,
            view_in_offset, view_valid_column, index,
            tmp_height, tmp_in_rowmajor, tmp_layer1out_rowmajor, tmp_layer2out_rowmajor,output_rowmajor,
            in_height, n1,  k1, activation1,
            n2, k2, activation2,
            n3, k3, activation3,
            weight1,  weight2, weight3,
            tmp_mlp_in, tmp_layer1out,
            tmp_layer2out, output);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

    m.def("expand_encoding_gemm_fp16_user_defined", &expand_encoding_gemm_fp16_user_defined, "Fused expand index + encoding + gemm based on cutlass",
    py::arg("app_latent"), py::arg("app_offset"),
    py::arg("features"), py::arg("viewdirs"), py::arg("n_freque_fea"), py::arg("n_freque_view"),
    py::arg("fea_encoding_offset"), py::arg("view_encoding_offset"), py::arg("fea_data_offset"), py::arg("view_data_offset"),
    py::arg("padding_offset"), py::arg("padding_width"),
    py::arg("view_in_offset"), py::arg("view_valid_column"), py::arg("index_view"),
    py::arg("tmp_m"), py::arg("tmp_in_rowmajor"), py::arg("tmp_layer1out_rowmajor"),
    py::arg("tmp_layer2out_rowmajor"), py::arg("output_rowmajor"),
    py::arg("m"), py::arg("n1"), py::arg("k1"), py::arg("activation1"),
    py::arg("n2"), py::arg("k2"), py::arg("activation2"),
    py::arg("n3"), py::arg("k3"), py::arg("activation3"),
    py::arg("weight1"),  py::arg("weight2"), py::arg("weight3"),
    py::arg("tmp_mlp_in"), py::arg("tmp_layer1out"),
    py::arg("tmp_layer2out"), py::arg("output"), py::arg("multiStream"));

    m.def("app_feature_expand_encoding_gemm_fp16_user_defined", &app_feature_expand_encoding_gemm_fp16_user_defined, "App_feature + Fused expand index + encoding + gemm based on cutlass",
    py::arg("plane_mul_line_T"), py::arg("basis_mat_weight"), py::arg("fea_tmp"), py::arg("gemm_app_n"), py::arg("gemm_app_k"),
    py::arg("gemm_app_activation"), py::arg("app_latent"), py::arg("app_offset"),
    py::arg("viewdirs"), py::arg("n_freque_fea"), py::arg("n_freque_view"),
    py::arg("fea_encoding_offset"), py::arg("view_encoding_offset"), py::arg("fea_data_offset"), py::arg("view_data_offset"),
    py::arg("padding_offset"), py::arg("padding_width"),
    py::arg("view_in_offset"), py::arg("view_valid_column"), py::arg("index_view"),
    py::arg("tmp_m"), py::arg("tmp_in_rowmajor"), py::arg("tmp_layer1out_rowmajor"),
    py::arg("tmp_layer2out_rowmajor"), py::arg("output_rowmajor"),
    py::arg("m"), py::arg("n1"), py::arg("k1"), py::arg("activation1"),
    py::arg("n2"), py::arg("k2"), py::arg("activation2"),
    py::arg("n3"), py::arg("k3"), py::arg("activation3"),
    py::arg("weight1"),  py::arg("weight2"), py::arg("weight3"),
    py::arg("tmp_mlp_in"), py::arg("tmp_layer1out"),
    py::arg("tmp_layer2out"), py::arg("output"), py::arg("multiStream"));

    m.def("app_feature_expand_encoding_gemm_fp16", &app_feature_expand_encoding_gemm_fp16_cpp, "App_feature + Fused expand index + encoding + gemm based on cutlass",
    py::arg("plane_mul_line_T"), py::arg("basis_mat_weight"), py::arg("fea_tmp"), py::arg("gemm_app_n"), py::arg("gemm_app_k"),
    py::arg("gemm_app_activation"), py::arg("app_latent"), py::arg("app_offset"),
    py::arg("viewdirs"), py::arg("n_freque_fea"), py::arg("n_freque_view"),
    py::arg("fea_encoding_offset"), py::arg("view_encoding_offset"), py::arg("fea_data_offset"), py::arg("view_data_offset"),
    py::arg("padding_offset"), py::arg("padding_width"),
    py::arg("view_in_offset"), py::arg("view_valid_column"), py::arg("index_view"),
    py::arg("tmp_m"), py::arg("tmp_in_rowmajor"), py::arg("tmp_layer1out_rowmajor"),
    py::arg("tmp_layer2out_rowmajor"), py::arg("output_rowmajor"),
    py::arg("m"), py::arg("n1"), py::arg("k1"), py::arg("activation1"),
    py::arg("n2"), py::arg("k2"), py::arg("activation2"),
    py::arg("n3"), py::arg("k3"), py::arg("activation3"),
    py::arg("weight1"),  py::arg("weight2"), py::arg("weight3"),
    py::arg("tmp_mlp_in"), py::arg("tmp_layer1out"),
    py::arg("tmp_layer2out"), py::arg("output"), py::arg("multiStream"));




}



////////////////////////////////////////////////////////////////////////////////
