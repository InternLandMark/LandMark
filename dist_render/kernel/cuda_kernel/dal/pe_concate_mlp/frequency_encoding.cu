#include  <stdexcept>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <mma.h>

#include <ATen/AccumulateType.h>

#include <assert.h>
#include <vector>
#include <torch/torch.h>


#include "common/index.h"
#include "common/helper.h"

#include "pe/frequency_encoding.h"

# define blockDIM 256
# define debug 0
# define PI 3.14159

void frequency_encoding_simple(torch::Tensor indata, torch::Tensor outdata,
        int n_freque, int in_offset, int in_valid_len, int out_offset){
        const int indata_width = indata.size(1);
        const int outdata_width = outdata.size(1);
        const int data_height = indata.size(0);

        if (indata.scalar_type() == torch::kHalf && outdata.scalar_type() == torch::kHalf) {
                const int num_elements = data_height*in_valid_len*n_freque*2;
                const int num_threads_row = in_valid_len*n_freque*2;
                int gridDIM =  (num_elements + blockDIM-1)/blockDIM;

                checkKernelErrors((frequency_encoding_simple_row_row<<<gridDIM, blockDIM>>>(
                num_elements, //所有执行的 threads的个数 行*列
                n_freque,
                indata_width,
                in_valid_len, // valid width of input data(width to be encoded))
                outdata_width, // height of pipelined output
                num_threads_row, // 处理一列数据所需线程
                in_offset,
                out_offset,
                reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
                reinterpret_cast<half*> (outdata.data_ptr<torch::Half>())
                )));

        }

}


void frequency_encoding(torch::Tensor indata, torch::Tensor outdata,
        int n_freque, int in_offset, int in_valid_len, int out_offset,
        int in_vector, int out_vector, bool is_inRowMajor, bool is_outRowMajor){

        const int indata_width = indata.size(1);
        const int outdata_width = outdata.size(1);
        const int data_height = indata.size(0);
     // Check if the tensor's data type is half
    if (indata.scalar_type() == torch::kHalf && outdata.scalar_type() == torch::kHalf) {
        if (is_inRowMajor && is_outRowMajor){
                if (in_vector==1 && out_vector==2 && n_freque == 2){
                        if (debug){
                                printf("encoding_half2half_row_row_in1_out2_freq2 \n");
                        }
                        int num_threads_row = in_valid_len;
                        uint64_t num_elements = data_height*(num_threads_row);
                        int gridDIM =  (num_elements + blockDIM-1)/blockDIM;

                        checkKernelErrors((encoding_half2half_row_row_in1_out2_freq2<<<gridDIM, blockDIM>>>(
                        num_elements, //所有执行的 threads的个数 行*列
                        n_freque,
                        indata_width,
                        in_valid_len, // valid width of input data(width to be encoded))
                        outdata_width, // height of pipelined output
                        num_threads_row,
                        in_offset,
                        out_offset,
                        reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
                        reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()))));
                        return;
                }



                if (in_vector==1 && out_vector==1 && n_freque == 2){
                        if (debug){
                                printf("encoding_half2half_row_row_in1_out1_freq2 \n");
                        }
                        int num_threads_row = in_valid_len;
                         uint64_t num_elements = data_height*(num_threads_row);
                        int gridDIM =  (num_elements + blockDIM-1)/blockDIM;

                        checkKernelErrors((encoding_half2half_row_row_in1_out1<2><<<gridDIM, blockDIM>>>(
                        num_elements, //所有执行的 threads的个数 行*列
                        indata_width,
                        in_valid_len, // valid width of input data(width to be encoded))
                        outdata_width, // height of pipelined output
                        num_threads_row, // 处理一列数据所需线程
                        in_offset,
                        out_offset,
                        reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
                        reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()))));

                        return;
                }
                if (in_vector==1 && out_vector==1 && n_freque == 3){
                        if (debug){
                                printf("encoding_half2half_row_row_in1_out1_freq3 \n");
                        }
                        int num_threads_row = in_valid_len;
                         uint64_t num_elements = data_height*(num_threads_row);
                        int gridDIM =  (num_elements + blockDIM-1)/blockDIM;

                        checkKernelErrors((encoding_half2half_row_row_in1_out1<3><<<gridDIM, blockDIM>>>(
                        num_elements, //所有执行的 threads的个数 行*列
                        indata_width,
                        in_valid_len, // valid width of input data(width to be encoded))
                        outdata_width, // height of pipelined output
                        num_threads_row, // 处理一列数据所需线程
                        in_offset,
                        out_offset,
                        reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
                        reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()))));

                        return;
                }

        }
        if (!is_inRowMajor && !is_outRowMajor){
                if (in_vector==2 && out_vector==2 && n_freque == 2){
                        if (debug){
                                printf("encoding_half2half_col_col_in2_out2 template \n");
                        }

                        int num_threads_col = data_height/2;
                        const uint64_t num_elements = (in_valid_len)*num_threads_col;
                        int gridDIM = (num_elements + blockDIM-1)/blockDIM;
                        // encoding
                        checkKernelErrors((encoding_half2half_col_col_in2_out2<2><<<gridDIM, blockDIM>>>
                        (
                        num_elements, //所有执行的 threads的个数 行*列
                        in_valid_len,
                        data_height,
                        num_threads_col, // 处理一列数据所需线程
                        reinterpret_cast<half2*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
                        reinterpret_cast<half2*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
                        )
                        ));
                        return;
                }

                if (in_vector==8 && out_vector==8 && n_freque == 2){

                        constexpr int alignment = 8;
                        int num_threads_col = data_height/alignment;
                        const uint64_t num_elements = (in_valid_len)*num_threads_col;
                        int gridDIM = (num_elements + blockDIM-1)/blockDIM;

                        if (debug){
                                printf("encoding_half2half_col_col_in8_out8 template, num_threads_col %d\n", num_threads_col);
                        }

                        // encoding
                        checkKernelErrors((encoding_half2half_col_col_in8_out8<half8, alignment, 2><<<gridDIM, blockDIM>>>
                        (
                        num_elements, //所有执行的 threads的个数 行*列
                        in_valid_len,
                        data_height,
                        num_threads_col, // 处理一列数据所需线程
                        reinterpret_cast<half8*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
                        reinterpret_cast<half8*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
                        )
                        ));
                        return;
                }

                if (in_vector==4 && out_vector==4 && n_freque == 2){

                        constexpr int alignment = 4;
                        int num_threads_col = data_height/alignment;
                        const uint64_t num_elements = (in_valid_len)*num_threads_col;
                        int gridDIM = (num_elements + blockDIM-1)/blockDIM;

                        if (debug){
                                printf("encoding_half2half_col_col_in4_out4 template, num_threads_col %d\n", num_threads_col);
                        }

                        // encoding
                        checkKernelErrors((encoding_half2half_col_col_in4_out4<half4, alignment, 2><<<gridDIM, blockDIM>>>
                        (
                        num_elements, //所有执行的 threads的个数 行*列
                        in_valid_len,
                        data_height,
                        num_threads_col, // 处理一列数据所需线程
                        reinterpret_cast<half4*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
                        reinterpret_cast<half4*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
                        )
                        ));
                        return;
                }

                if (in_vector==2 && out_vector==2 && n_freque == 3){
                        if (debug){
                                printf("encoding_half2half_col_col_in2_out2 freq3 template \n");
                        }

                        int num_threads_col = data_height/2;
                        const uint64_t num_elements = (in_valid_len)*num_threads_col;
                        int gridDIM = (num_elements + blockDIM-1)/blockDIM;
                        // encoding
                        checkKernelErrors((encoding_half2half_col_col_in2_out2<3><<<gridDIM, blockDIM>>>
                        (
                        num_elements, //所有执行的 threads的个数 行*列
                        in_valid_len,
                        data_height,
                        num_threads_col, // 处理一列数据所需线程
                        reinterpret_cast<half2*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
                        reinterpret_cast<half2*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
                        )
                        ));
                        return;
                }

                if (in_vector==1 && out_vector==2 && n_freque == 2){
                        if (debug){
                                printf("encoding_half2half_col_col_in1_out2_freq2 \n");
                        }
                        int num_threads_col = data_height/2;
                        const uint64_t num_elements = (in_valid_len)*num_threads_col;
                        int gridDIM = (num_elements + blockDIM-1)/blockDIM;
                        // encoding
                        checkKernelErrors((encoding_half2half_col_col_in1_out2<2><<<gridDIM, blockDIM>>>
                                (
                                num_elements, //所有执行的 threads的个数 行*列
                                in_valid_len,
                                data_height,
                                num_threads_col, // 处理一列数据所需线程
                                reinterpret_cast<half*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
                                reinterpret_cast<half2*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
                                )
                        ));
                        return;
                }

                if (in_vector==1 && out_vector==2 && n_freque == 3){
                        if (debug){
                                printf("encoding_half2half_col_col_in1_out2_freq3 \n");
                        }
                        int num_threads_col = data_height/2;
                        const uint64_t num_elements = (in_valid_len)*num_threads_col;
                        int gridDIM = (num_elements + blockDIM-1)/blockDIM;
                        // encoding
                        checkKernelErrors((encoding_half2half_col_col_in1_out2<3><<<gridDIM, blockDIM>>>
                                (
                                num_elements, //所有执行的 threads的个数 行*列
                                in_valid_len,
                                data_height,
                                num_threads_col, // 处理一列数据所需线程
                                reinterpret_cast<half*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
                                reinterpret_cast<half2*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
                                )
                        ));
                        return;
                }

                if (n_freque == 2) {
                        if (debug){
                                printf("encoding_half2half_col_col_in1_out1_freq2 \n");
                        }
                        int num_threads_col = data_height;
                        const uint64_t num_elements = (in_valid_len)*num_threads_col;
                        int gridDIM = (num_elements + blockDIM-1)/blockDIM;
                        // encoding
                        checkKernelErrors((encoding_half2half_col_col_in1_out1<2><<<gridDIM, blockDIM>>>
                        (
                        num_elements, //所有执行的 threads的个数 行*列
                        in_valid_len,
                        data_height,
                        num_threads_col, // 处理一列数据所需线程
                        reinterpret_cast<half*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
                        reinterpret_cast<half*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
                        )
                        ));
                        return;
                }

                if (n_freque == 3) {
                        if (debug){
                                printf("encoding_half2half_col_col_in1_out1_freq3 \n");
                        }
                        int num_threads_col = data_height;
                        const uint64_t num_elements = (in_valid_len)*num_threads_col;
                        int gridDIM = (num_elements + blockDIM-1)/blockDIM;
                        // encoding
                        checkKernelErrors((encoding_half2half_col_col_in1_out1<3><<<gridDIM, blockDIM>>>
                        (
                        num_elements, //所有执行的 threads的个数 行*列
                        in_valid_len,
                        data_height,
                        num_threads_col, // 处理一列数据所需线程
                        reinterpret_cast<half*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
                        reinterpret_cast<half*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
                        )
                        ));
                        return;
                }

        }

    }


}

void frequency_encoding_mvdata(torch::Tensor indata, torch::Tensor outdata,
        int n_freque, int in_read_offset, int in_valid_len, int in_write_offset, int out_offset,
        int outdata_width,
        int in_vector, int out_vector, bool is_inRowMajor, bool is_outRowMajor){

  const int indata_width = indata.size(1);
  const int data_height = indata.size(0);


  if (indata.scalar_type() == torch::kHalf && outdata.scalar_type() == torch::kHalf) {
        if (is_inRowMajor && is_outRowMajor){
                if (in_vector==1 && out_vector==2 && n_freque == 2){
                        if (debug){
                                printf("encoding_half2half_row_row_in1_out2_freq2_mv_data \n");
                        }
                        int num_threads_row = in_valid_len;
                        uint64_t num_elements = data_height*(num_threads_row);
                        int gridDIM =  (num_elements + blockDIM-1)/blockDIM;

                        checkKernelErrors((encoding_half2half_row_row_in1_out2_freq2_mv_data<<<gridDIM, blockDIM>>>(
                        num_elements, //所有执行的 threads的个数 行*列
                        n_freque,
                        indata_width,
                        in_valid_len, // valid width of input data(width to be encoded))
                        outdata_width, // height of pipelined output
                        num_threads_row,
                        in_read_offset,
                        in_write_offset,
                        out_offset,
                        reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
                        reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()))));
                        return;
                }

                if (in_vector==1 && out_vector==1 && n_freque == 2){
                        if (debug){
                                printf("encoding_half2half_row_row_in1_out1_mv_data \n");
                        }
                        int num_threads_row = in_valid_len;
                         uint64_t num_elements = data_height*(num_threads_row);
                        int gridDIM =  (num_elements + blockDIM-1)/blockDIM;

                        checkKernelErrors((encoding_half2half_row_row_in1_out1_mv_data<2><<<gridDIM, blockDIM>>>(
                        num_elements, //所有执行的 threads的个数 行*列
                        indata_width,
                        in_valid_len, // valid width of input data(width to be encoded))
                        outdata_width, // height of pipelined output
                        num_threads_row, // 处理一列数据所需线程
                        in_read_offset,
                        in_write_offset,
                        out_offset,
                        reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
                        reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()))));

                        return;
                }
                if (in_vector==1 && out_vector==1 && n_freque == 3){
                        if (debug){
                                printf("encoding_half2half_row_row_in1_out1_mv_data freq3 \n");
                        }
                        int num_threads_row = in_valid_len;
                         uint64_t num_elements = data_height*(num_threads_row);
                        int gridDIM =  (num_elements + blockDIM-1)/blockDIM;

                        checkKernelErrors((encoding_half2half_row_row_in1_out1_mv_data<3><<<gridDIM, blockDIM>>>(
                        num_elements, //所有执行的 threads的个数 行*列
                        indata_width,
                        in_valid_len, // valid width of input data(width to be encoded))
                        outdata_width, // height of pipelined output
                        num_threads_row, // 处理一列数据所需线程
                        in_read_offset,
                        in_write_offset,
                        out_offset,
                        reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
                        reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()))));

                        return;
                }

        }
        // if (!is_inRowMajor && !is_outRowMajor){
        //         if (in_vector==2 && out_vector==2 && n_freque == 2){
        //                 if (debug){
        //                         printf("encoding_half2half_col_col_in2_out2 template \n");
        //                 }

        //                 int num_threads_col = data_height/2;
        //                 const uint64_t num_elements = (in_valid_len)*num_threads_col;
        //                 int gridDIM = (num_elements + blockDIM-1)/blockDIM;
        //                 // encoding
        //                 checkKernelErrors((encoding_half2half_col_col_in2_out2<2><<<gridDIM, blockDIM>>>
        //                 (
        //                 num_elements, //所有执行的 threads的个数 行*列
        //                 in_valid_len,
        //                 data_height,
        //                 num_threads_col, // 处理一列数据所需线程
        //                 reinterpret_cast<half2*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
        //                 reinterpret_cast<half2*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
        //                 )
        //                 ));
        //                 return;
        //         }

        //         if (in_vector==8 && out_vector==8 && n_freque == 2){

        //                 constexpr int alignment = 8;
        //                 int num_threads_col = data_height/alignment;
        //                 const uint64_t num_elements = (in_valid_len)*num_threads_col;
        //                 int gridDIM = (num_elements + blockDIM-1)/blockDIM;

        //                 if (debug){
        //                         printf("encoding_half2half_col_col_in8_out8 template, num_threads_col %d\n", num_threads_col);
        //                 }

        //                 // encoding
        //                 checkKernelErrors((encoding_half2half_col_col_in8_out8<half8, alignment, 2><<<gridDIM, blockDIM>>>
        //                 (
        //                 num_elements, //所有执行的 threads的个数 行*列
        //                 in_valid_len,
        //                 data_height,
        //                 num_threads_col, // 处理一列数据所需线程
        //                 reinterpret_cast<half8*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
        //                 reinterpret_cast<half8*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
        //                 )
        //                 ));
        //                 return;
        //         }

        //         if (in_vector==4 && out_vector==4 && n_freque == 2){

        //                 constexpr int alignment = 4;
        //                 int num_threads_col = data_height/alignment;
        //                 const uint64_t num_elements = (in_valid_len)*num_threads_col;
        //                 int gridDIM = (num_elements + blockDIM-1)/blockDIM;

        //                 if (debug){
        //                         printf("encoding_half2half_col_col_in4_out4 template, num_threads_col %d\n", num_threads_col);
        //                 }

        //                 // encoding
        //                 checkKernelErrors((encoding_half2half_col_col_in4_out4<half4, alignment, 2><<<gridDIM, blockDIM>>>
        //                 (
        //                 num_elements, //所有执行的 threads的个数 行*列
        //                 in_valid_len,
        //                 data_height,
        //                 num_threads_col, // 处理一列数据所需线程
        //                 reinterpret_cast<half4*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
        //                 reinterpret_cast<half4*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
        //                 )
        //                 ));
        //                 return;
        //         }

        //         if (in_vector==2 && out_vector==2 && n_freque == 3){
        //                 if (debug){
        //                         printf("encoding_half2half_col_col_in2_out2 freq3 template \n");
        //                 }

        //                 int num_threads_col = data_height/2;
        //                 const uint64_t num_elements = (in_valid_len)*num_threads_col;
        //                 int gridDIM = (num_elements + blockDIM-1)/blockDIM;
        //                 // encoding
        //                 checkKernelErrors((encoding_half2half_col_col_in2_out2<3><<<gridDIM, blockDIM>>>
        //                 (
        //                 num_elements, //所有执行的 threads的个数 行*列
        //                 in_valid_len,
        //                 data_height,
        //                 num_threads_col, // 处理一列数据所需线程
        //                 reinterpret_cast<half2*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
        //                 reinterpret_cast<half2*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
        //                 )
        //                 ));
        //                 return;
        //         }

        //         if (in_vector==1 && out_vector==2 && n_freque == 2){
        //                 if (debug){
        //                         printf("encoding_half2half_col_col_in1_out2_freq2 \n");
        //                 }
        //                 int num_threads_col = data_height/2;
        //                 const uint64_t num_elements = (in_valid_len)*num_threads_col;
        //                 int gridDIM = (num_elements + blockDIM-1)/blockDIM;
        //                 // encoding
        //                 checkKernelErrors((encoding_half2half_col_col_in1_out2<2><<<gridDIM, blockDIM>>>
        //                         (
        //                         num_elements, //所有执行的 threads的个数 行*列
        //                         in_valid_len,
        //                         data_height,
        //                         num_threads_col, // 处理一列数据所需线程
        //                         reinterpret_cast<half*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
        //                         reinterpret_cast<half2*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
        //                         )
        //                 ));
        //                 return;
        //         }

        //         if (in_vector==1 && out_vector==2 && n_freque == 3){
        //                 if (debug){
        //                         printf("encoding_half2half_col_col_in1_out2_freq3 \n");
        //                 }
        //                 int num_threads_col = data_height/2;
        //                 const uint64_t num_elements = (in_valid_len)*num_threads_col;
        //                 int gridDIM = (num_elements + blockDIM-1)/blockDIM;
        //                 // encoding
        //                 checkKernelErrors((encoding_half2half_col_col_in1_out2<3><<<gridDIM, blockDIM>>>
        //                         (
        //                         num_elements, //所有执行的 threads的个数 行*列
        //                         in_valid_len,
        //                         data_height,
        //                         num_threads_col, // 处理一列数据所需线程
        //                         reinterpret_cast<half*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
        //                         reinterpret_cast<half2*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
        //                         )
        //                 ));
        //                 return;
        //         }

        //         if (n_freque == 2) {
        //                 if (debug){
        //                         printf("encoding_half2half_col_col_in1_out1_freq2 \n");
        //                 }
        //                 int num_threads_col = data_height;
        //                 const uint64_t num_elements = (in_valid_len)*num_threads_col;
        //                 int gridDIM = (num_elements + blockDIM-1)/blockDIM;
        //                 // encoding
        //                 checkKernelErrors((encoding_half2half_col_col_in1_out1<2><<<gridDIM, blockDIM>>>
        //                 (
        //                 num_elements, //所有执行的 threads的个数 行*列
        //                 in_valid_len,
        //                 data_height,
        //                 num_threads_col, // 处理一列数据所需线程
        //                 reinterpret_cast<half*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
        //                 reinterpret_cast<half*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
        //                 )
        //                 ));
        //                 return;
        //         }

        //         if (n_freque == 3) {
        //                 if (debug){
        //                         printf("encoding_half2half_col_col_in1_out1_freq3 \n");
        //                 }
        //                 int num_threads_col = data_height;
        //                 const uint64_t num_elements = (in_valid_len)*num_threads_col;
        //                 int gridDIM = (num_elements + blockDIM-1)/blockDIM;
        //                 // encoding
        //                 checkKernelErrors((encoding_half2half_col_col_in1_out1<3><<<gridDIM, blockDIM>>>
        //                 (
        //                 num_elements, //所有执行的 threads的个数 行*列
        //                 in_valid_len,
        //                 data_height,
        //                 num_threads_col, // 处理一列数据所需线程
        //                 reinterpret_cast<half*> (indata.data_ptr<torch::Half>() + in_offset*data_height),
        //                 reinterpret_cast<half*> (outdata.data_ptr<torch::Half>() + out_offset*data_height)
        //                 )
        //                 ));
        //                 return;
        //         }

        // }

    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &frequency_encoding, "frequency_encoding (CUDA)",
        py::arg("indata"),  py::arg("outdata"),
        py::arg("n_freque"), py::arg("in_offset"), py::arg("in_valid_len"), py::arg("out_offset"),
        py::arg("in_vector"), py::arg("out_vector"), py::arg("is_inRowMajor"), py::arg("is_outRowMajor"));

  m.def("pe_mvdata", &frequency_encoding_mvdata, "frequency_encoding_mvdata (CUDA)",
        py::arg("indata"),  py::arg("outdata"),
        py::arg("n_freque"), py::arg("in_read_offset"), py::arg("in_valid_len"), py::arg("in_write_offset"), py::arg("out_offset"),
        py::arg("outdata_width"),
        py::arg("in_vector"), py::arg("out_vector"), py::arg("is_inRowMajor"), py::arg("is_outRowMajor"));

  m.def("simple", &frequency_encoding_simple, "frequency_encoding (CUDA)",
        py::arg("indata"),  py::arg("outdata"),
        py::arg("n_freque"), py::arg("in_offset"), py::arg("in_valid_len"), py::arg("out_offset"));
}
