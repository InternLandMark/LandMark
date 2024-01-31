#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA funciton declearition
void AssignBlocksToSamples_cuda(torch::Tensor masks, torch::Tensor xyz_sample_valid, torch::Tensor output, int valid_num, int plane_x, int plane_y, int blockdim);
void AssignBlocksToSamplesNorm_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y);
void AssignBlocksToSamplesNormInt_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y, int sampleray_blockdim, int sampleray_samplesperblock);
void AssignBlocksToSamplesNormInt_relative_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y, int corner_block_idx_x, int corner_block_idx_y, int neighbour_width, int sampleray_blockdim, int sampleray_samplesperblock);

void AssignBlocksToSamplesNormInt_fused_relative_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, int plane_x, int plane_y, int corner_block_idx_x, int corner_block_idx_y, int neighbour_width, int blockdim, int samplesperblock,
std::vector<torch::Tensor> plane, std::vector<torch::Tensor> line,
torch::Tensor hw_in, torch::Tensor plane_line_ptr, torch::Tensor sigma);// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void assign_blocks_to_samples(torch::Tensor masks, torch::Tensor xyz_sample_valid, torch::Tensor output, int valid_num, int plane_x, int plane_y, bool debug_mode, int blockdim)
{
    // CHECK_INPUT(masks);
    CHECK_INPUT(xyz_sample_valid);
    CHECK_INPUT(output);

    int iter_times = 102;
    int count_times = 100;


    if (debug_mode == true)
    {
        cudaEvent_t start, stop;
        float time_avg = 0.0;
        float elapsedTime = 0.0;

        time_avg = 0.0;
        for (int i = 0; i < iter_times; i++)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();
            cudaEventRecord(start, 0);
            AssignBlocksToSamples_cuda(masks, xyz_sample_valid, output, valid_num, plane_x, plane_y, blockdim);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            if (i >= (iter_times - count_times))
                time_avg += elapsedTime / count_times;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        std::cout << "AssignBlocksToSamples time: " << time_avg << "ms" << std::endl;
    }
    else
    {
        AssignBlocksToSamples_cuda(masks, xyz_sample_valid, output, valid_num, plane_x, plane_y, blockdim);
    }
}

void assign_blocks_to_samples_norm(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y, bool int_mode, bool debug_mode, int sampleray_blockdim, int sampleray_samplesperblock)
{
    CHECK_INPUT(ray_valid);
    CHECK_INPUT(xyz_sample);
    int iter_times = 102;
    int count_times = 100;


    if (debug_mode == true)
    {
        cudaEvent_t start, stop;
        float time_avg = 0.0;
        float elapsedTime = 0.0;

        time_avg = 0.0;
        for (int i = 0; i < iter_times; i++)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();
            cudaEventRecord(start, 0);
            if (int_mode == true)
                AssignBlocksToSamplesNormInt_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y, sampleray_blockdim, sampleray_samplesperblock);
            else
                AssignBlocksToSamplesNorm_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            if (i >= (iter_times - count_times))
                time_avg += elapsedTime / count_times;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        std::cout << "AssignBlocksToSamples time: " << time_avg << "ms" << std::endl;
    }
    else
    {
        if (int_mode == true)
            AssignBlocksToSamplesNormInt_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y, sampleray_blockdim, sampleray_samplesperblock);
        else
            AssignBlocksToSamplesNorm_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y);
    }
}

void assign_blocks_to_samples_norm_relative(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks,
int plane_x, int plane_y, int corner_block_idx_x, int corner_block_idx_y, int neighbour_width,
bool int_mode, bool debug_mode, int sampleray_blockdim, int sampleray_samplesperblock)
{
    CHECK_INPUT(ray_valid);
    CHECK_INPUT(xyz_sample);
    int iter_times = 102;
    int count_times = 100;


    if (debug_mode == true)
    {
        cudaEvent_t start, stop;
        float time_avg = 0.0;
        float elapsedTime = 0.0;

        time_avg = 0.0;
        for (int i = 0; i < iter_times; i++)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();
            cudaEventRecord(start, 0);
            if (int_mode == true)
                AssignBlocksToSamplesNormInt_relative_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y, corner_block_idx_x, corner_block_idx_y, neighbour_width, sampleray_blockdim, sampleray_samplesperblock);
            else
                AssignBlocksToSamplesNorm_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            if (i >= (iter_times - count_times))
                time_avg += elapsedTime / count_times;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        std::cout << "AssignBlocksToSamples time: " << time_avg << "ms" << std::endl;
    }
    else
    {
        if (int_mode == true)
            AssignBlocksToSamplesNormInt_relative_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y, corner_block_idx_x, corner_block_idx_y, neighbour_width, sampleray_blockdim, sampleray_samplesperblock);
        else
            AssignBlocksToSamplesNorm_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y);
    }
}

void assign_blocks_to_samples_norm_fused_relative(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample,
int plane_x, int plane_y, int corner_block_idx_x, int corner_block_idx_y, int neighbour_width,
bool int_mode, bool debug_mode, int sampleray_blockdim, int sampleray_samplesperblock,
std::vector<torch::Tensor> plane_vec, std::vector<torch::Tensor> line_vec,
torch::Tensor hw_in,
torch::Tensor plane_line_ptr,
torch::Tensor sigma)
{
    CHECK_INPUT(ray_valid);
    CHECK_INPUT(xyz_sample);
    int iter_times = 102;
    int count_times = 100;


    if (debug_mode == true)
    {
        cudaEvent_t start, stop;
        float time_avg = 0.0;
        float elapsedTime = 0.0;

        time_avg = 0.0;
        for (int i = 0; i < iter_times; i++)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();
            cudaEventRecord(start, 0);
            if (int_mode == true)
                AssignBlocksToSamplesNormInt_fused_relative_cuda(xyz_sample, ray_valid, b_sample, plane_x, plane_y,
                corner_block_idx_x, corner_block_idx_y, neighbour_width, sampleray_blockdim, sampleray_samplesperblock, plane_vec, line_vec, hw_in, plane_line_ptr, sigma);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            if (i >= (iter_times - count_times))
                time_avg += elapsedTime / count_times;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        std::cout << "AssignBlocksToSamples time: " << time_avg << "ms" << std::endl;
    }
    else
    {
        if (int_mode == true)
            AssignBlocksToSamplesNormInt_fused_relative_cuda(xyz_sample, ray_valid, b_sample, plane_x, plane_y,
            corner_block_idx_x, corner_block_idx_y, neighbour_width,  sampleray_blockdim, sampleray_samplesperblock, plane_vec, line_vec, hw_in, plane_line_ptr, sigma);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cuda_ori", &assign_blocks_to_samples, "assign_blocks_to_samples (CUDA)");
    m.def("cuda_norm", &assign_blocks_to_samples_norm, "assign_blocks_to_samples_norm (CUDA)");
    m.def("cuda_norm_relative", &assign_blocks_to_samples_norm_relative, "assign_blocks_to_samples_norm (CUDA)");
    m.def("cuda_fused_relative", &assign_blocks_to_samples_norm_fused_relative, "assign_blocks_to_samples_norm (CUDA)");
}
