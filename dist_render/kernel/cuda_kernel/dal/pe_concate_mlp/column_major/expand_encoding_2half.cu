# include "index/expand_index_column_major.h"
# include "pe/column_major/pe_column_major_half2half.h"

# define debug 0

// 任意高度, 任意frequency
__global__ void expand_index_encoding_2half(
        const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t outdata_height,
        const uint32_t indata_width,
        const uint32_t indata_valid_width,
        const int num_threads_col,
        const int in_offset,
        const int n_frequencies,
        half* indata_ptr,
        int* index,
        half* outdata
        )
{
        const uint64_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (encoded_index >= num_elements) return;
        /*
        相邻的线程处理同一列
        */
        const uint32_t i =  encoded_index % num_threads_col;
        const uint32_t j = encoded_index / num_threads_col;


        half indata = expand_index_half2half(i,  j,
                indata_width,
                in_offset,
                indata_ptr,
                index
                );
        frequency_encoding_device(
        i,  j, encoded_index, indata, n_frequencies, indata_valid_width, outdata_height,
        outdata);
}

// 高度是单数, frequency == 2
__global__ void expand_index_encoding_freq2_2half(
        const uint32_t num_elements, //所有执行的 threads的个数 行*列
        const uint32_t outdata_height,
        const uint32_t indata_width,
        const uint32_t indata_valid_width,
        const int num_threads_col,
        const int in_offset,
        const int n_frequencies,
        half* indata_ptr,
        int* index,
        half* outdata_expand, // pointer to rewrite data
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


        half indata = expand_index_half2half(i,  j,
                indata_width,
                in_offset,
                indata_ptr,
                index
                );

        outdata_expand[IDX2C(i, j, outdata_height)]  = 	indata;
        frequency_encoding_device_freq2(
        i,  j,  indata, n_frequencies, indata_valid_width, outdata_height,
        outdata_encoding);
}

// 高度是复数, frequency == 2
__global__ void expand_index_encoding_2half_2x_freq2(
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



/*
indata: sampled input data (after app_mask)
index: a one dimensional index which reflect from input to output
outdata: output data (restore from app_mask)
in_offset: the first column that we read the data;
out_offset: the first column that we rewrite the data to output
in_valid_len: the number of columns of data we want to encode
encoding_offset: the first column of output data that the encoding data reside in
n_freq : frequency of encoding
*/

void expand_index(torch::Tensor indata, torch::Tensor index, torch::Tensor outdata,
int in_offset, int in_valid_len,
int out_offset, int encoding_offset, int n_freq){
        /*
        indata shape N1 *3
        outdata shape N2 * 153
        index shape N2
        */
        const int indata_height = indata.size(0);
        const int outdata_height = outdata.size(0);
        const int indata_width = indata.size(1);
        const int outdata_width = outdata.size(1);

    if (outdata_height%2 == 1 && n_freq ==2){
        const int num_threads_col = outdata_height;
        const int num_elements = (in_valid_len)*num_threads_col;
        int gridDIM = (num_elements + blockDIM-1)/blockDIM;

        if (debug){
                printf("num_elements %d \n", num_elements);
                printf("use kernel expand_index_encoding_freq2_2half");
        }
        expand_index_encoding_freq2_2half<<<gridDIM, blockDIM>>>
          (num_elements,outdata_height, indata_width, in_valid_len, num_threads_col,
          in_offset,  n_freq,
                reinterpret_cast<half*> (indata.to(at::kHalf).data_ptr<torch::Half>()),
                index.data_ptr<int>(),
                reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()+ out_offset*outdata_height),
                reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()+ encoding_offset*outdata_height)
                );
        }
    else if (outdata_height%2 == 0 && n_freq ==2){
        const int num_threads_col = outdata_height/2;
        const int num_elements = (in_valid_len)*num_threads_col;
        int gridDIM = (num_elements + blockDIM-1)/blockDIM;

        if (debug){
                printf("num_elements %d \n", num_elements);
                printf("use expand_index_encoding_2half_2x_freq2");
        }

        expand_index_encoding_2half_2x_freq2<<<gridDIM, blockDIM>>>
          (num_elements,outdata_height, indata_width, in_valid_len, num_threads_col,
          in_offset,  n_freq,
                reinterpret_cast<half*> (indata.to(at::kHalf).data_ptr<torch::Half>()),
                index.data_ptr<int>(),
                reinterpret_cast<half2*> (outdata.data_ptr<torch::Half>()+ out_offset*outdata_height),
                reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()+ encoding_offset*outdata_height)
                );
    }
    else{
      std::cerr << "Does not capable of caculating when encoding frequecy != 2." << std::endl;
//       const int num_threads_col = outdata_height;
//       const int num_elements = (in_valid_len*2*n_freq)*num_threads_col;
//       int gridDIM = (num_elements + blockDIM-1)/blockDIM;

//       if (debug){
//                 printf("num_elements %d \n", num_elements);
//                 printf("use expand_index_encoding_2half");
//         }

//       expand_index_encoding_2half<<<gridDIM, blockDIM>>>
//          (num_elements,outdata_height, indata_width, in_valid_len, num_threads_col, in_offset, n_freq,
//                 reinterpret_cast<half*> (indata.to(at::kHalf).data_ptr<torch::Half>()),
//                 index.data_ptr<int>(),
//                 reinterpret_cast<half*> (outdata.data_ptr<torch::Half>()+ encoding_offset*outdata_height)
//                 );
      }


    getLastCudaError("Frequency_encoding execution failed\n");

    return;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &expand_index, "expand_index row major to column major (CUDA)",
  py::arg("indata"), py::arg("index"), py::arg("outdata"),
  py::arg("in_offset"), py::arg("in_valid_len"),
  py::arg("out_offset"), py::arg("encoding_offset"), py::arg("n_freq"));
}
