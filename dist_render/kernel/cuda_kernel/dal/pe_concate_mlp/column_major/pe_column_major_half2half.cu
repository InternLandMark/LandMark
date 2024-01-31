# include "pe/column_major/pe_column_major_half2half.h"
# define debug 0

__global__ void frequency_encoding_kernel_freq2(
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
        const uint32_t j = encoded_index / num_threads_col; //输入的列 = 输出的列/4

        /*
        sin                          cos
        freq0 freq1 ...........      freq0 freq1 ...........
        =======================      =======================

        */
       frequency_encoding_device_freq2(
        i, //输出的行
        j, //输入的列 = 输出的列/4
        num_elements, //所有执行的 threads的个数 行*列
        n_frequencies,
        offset, // 30 for features
        indata_width,
        indata_height,
        outdata_width,
        outdata_height,
        num_threads_col, // 处理一列数据所需线程
        data_in,
        data_out);
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
        freq0 freq1 frea2 ...........      freq0 freq1 frea2 ...........
        =======================      =======================

        */
       frequency_encoding_device(
        i, //输出的行
        j, //输出的列
        encoded_index,
        num_elements, //所有执行的 threads的个数 行*列
        n_frequencies,
        offset, // 30 for features
        indata_width,
        indata_height,
        outdata_width,
        outdata_height,
        num_threads_col, // 处理一列数据所需线程
        data_in,
        data_out);
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

        frequency_encoding_device_2x_freq2
        (i, j,
         num_elements, //所有执行的 threads的个数 行*列
        n_frequencies,
        offset, // 30 for features
        indata_width,
        indata_height,
        outdata_width,
        outdata_height,
        num_threads_col, // 处理一列数据所需线程
        data_in,
        data_out);

}

/*
indata: sampled input data (after app_mask)
outdata: output data (restore from app_mask)
n_freque : frequency of encoding
offset: the first column of output data that the encoding data reside in
*/

void frequency_encoding(torch::Tensor indata, torch::Tensor outdata, int n_freque, int offset){

    const int indata_height = indata.size(0);
    const int outdata_height = outdata.size(0);
    const int indata_width = indata.size(1);
    const int outdata_width = outdata.size(1);


    if (indata_height%2 == 1 && n_freque ==2){
        const int num_threads_col = outdata_height;
        const int num_elements = (indata_width)*num_threads_col;
        int gridDIM = (num_elements + blockDIM-1)/blockDIM;
        frequency_encoding_kernel_freq2<<<gridDIM, blockDIM>>>
          (num_elements,
            n_freque,
            offset,
            indata_width,
            indata_height,
            outdata_width,
            outdata_height,
            num_threads_col,
            reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
            reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()+ offset*outdata_height)
        );
        }
    else if (indata_height%2 == 0 && n_freque ==2){
        const int num_threads_col = outdata_height/2;
        const int num_elements = (indata_width)*num_threads_col;
        int gridDIM = (num_elements + blockDIM-1)/blockDIM;

        frequency_encoding_kernel_2x_freq2<<<gridDIM, blockDIM>>>
          (num_elements,
            n_freque,
            offset,
            indata_width,
            indata_height,
            outdata_width,
            outdata_height,
            num_threads_col,
            reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
            reinterpret_cast<half*> (outdata.data_ptr<torch::Half>() + offset*outdata_height)
          );
    }
    else{

      const int num_threads_col = outdata_height;
      const int num_elements = (indata_width*2*n_freque)*num_threads_col;
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
          reinterpret_cast<half*> (indata.data_ptr<torch::Half>()),
          reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()+ offset*outdata_height)
      );
      }


    getLastCudaError("Frequency_encoding execution failed\n");

    return;

}

/*
indata: sampled input data (after app_mask)
outdata: output data (restore from app_mask)
n_freque : frequency of encoding
offset: the first column of output data that the encoding data reside in
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &frequency_encoding, "frequency_encoding (CUDA)",
   py::arg("indata"),  py::arg("outdata"),
  py::arg("n_freque"), py::arg("offset"));
}
