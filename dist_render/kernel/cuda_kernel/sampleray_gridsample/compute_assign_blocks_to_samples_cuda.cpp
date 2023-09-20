#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA funciton declearition
void AssignBlocksToSamples_cuda(torch::Tensor masks, torch::Tensor xyz_sample_valid, torch::Tensor output, int valid_num, int plane_x, int plane_y, int blockdim);
void AssignBlocksToSamplesNorm_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y);
void AssignBlocksToSamplesNormInt_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y);
// C++ interface

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

void assign_blocks_to_samples_norm(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y, bool int_mode, bool debug_mode)
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
                AssignBlocksToSamplesNormInt_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y);
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
            AssignBlocksToSamplesNormInt_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y);
        else
            AssignBlocksToSamplesNorm_cuda(xyz_sample, ray_valid, b_sample, masks, plane_x, plane_y);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cuda_ori", &assign_blocks_to_samples, "assign_blocks_to_samples (CUDA)");
    m.def("cuda_norm", &assign_blocks_to_samples_norm, "assign_blocks_to_samples_norm (CUDA)");
}
