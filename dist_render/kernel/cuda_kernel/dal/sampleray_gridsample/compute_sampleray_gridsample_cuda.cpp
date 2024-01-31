#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA funciton declearition
void GridSample3D_2d_bool_dev_cuda(torch::Tensor input, torch::Tensor grid, torch::Tensor output, torch::Tensor aabb, torch::Tensor alphaMask_aabb, int valid_samples_num, float alpha_thre, int blockdim);
void SampleRay_WithinHull_coord_trunc_precaltmin_validsamples_cuda_opt(torch::Tensor rays_chunk, torch::Tensor aabb, torch::Tensor near_far, bool is_train, int N_samples, int valid_samples_num, torch::Tensor rays_pts, torch::Tensor z_vals, torch::Tensor mask_outbbox, int sampleray_blockdim, int sampleray_samplesperblock);
void CalTValMin_cuda(torch::Tensor rays_chunk, torch::Tensor aabb, torch::Tensor near_far, torch::Tensor t_val_min);
void CalTValMin_Sharedmem_cuda(torch::Tensor rays_chunk, torch::Tensor aabb, torch::Tensor near_far, torch::Tensor t_val_min, torch::Tensor tval_min_index, int N_samples, float* tval_min_min, bool simple_mode);
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

int sampleray_gridsample(torch::Tensor rays_chunk, torch::Tensor aabb, torch::Tensor alphaMask_aabb,
torch::Tensor near_far, bool is_train, int N_samples,
torch::Tensor valid_samples_num_tensor, //int valid_samples_num,
torch::Tensor rays_pts, torch::Tensor z_vals, torch::Tensor mask_outbbox, torch::Tensor tval_min, torch::Tensor tval_min_index, torch::Tensor density_plane_line_sum, torch::Tensor sigma_feature_cuda, float alpha_thre, bool simple_mode, bool debug_mode, int sampleray_blockdim, int sampleray_samplesperblock, int gridsample_blockdim)
{
    CHECK_INPUT(rays_chunk);
    CHECK_INPUT(aabb);
    CHECK_INPUT(near_far);
    CHECK_INPUT(rays_pts);
    CHECK_INPUT(z_vals);
    CHECK_INPUT(mask_outbbox);
    CHECK_INPUT(tval_min);
    CHECK_INPUT(density_plane_line_sum);
    CHECK_INPUT(sigma_feature_cuda);

    int iter_times = 1; //102;
    int count_times = 1; //100;

    int valid_samples_num = valid_samples_num_tensor.index({0}).item<int>();// int valid_samples_num;
    float tval_min_min;

    if (debug_mode == true)
    {
        cudaEvent_t start, stop;
        float time_avg = 0.0;
        float elapsedTime = 0.0;

        for (int i = 0; i < iter_times; i++)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();
            cudaEventRecord(start, 0);
            CalTValMin_Sharedmem_cuda(rays_chunk, aabb, near_far, tval_min, tval_min_index, N_samples, &tval_min_min, simple_mode);
            valid_samples_num = N_samples - ceil(tval_min_min * N_samples);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            if (i >= (iter_times - count_times))
                time_avg += elapsedTime / count_times;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        std::cout << "valid_samples_num: " << valid_samples_num << std::endl;

        std::cout << "CalTValMin time: " << time_avg << "ms" << std::endl;

        time_avg = 0.0;
        for (int i = 0; i < iter_times; i++)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();
            cudaEventRecord(start, 0);
            SampleRay_WithinHull_coord_trunc_precaltmin_validsamples_cuda_opt(rays_chunk, aabb, near_far, is_train, N_samples, valid_samples_num, rays_pts, z_vals, mask_outbbox, sampleray_blockdim, sampleray_samplesperblock);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            if (i >= (iter_times - count_times))
                time_avg += elapsedTime / count_times;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        std::cout << "SampleRay time: " << time_avg << "ms" << std::endl;

        time_avg = 0.0;
        for (int i = 0; i < iter_times; i++)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();
            cudaEventRecord(start, 0);
            GridSample3D_2d_bool_dev_cuda(density_plane_line_sum, rays_pts, sigma_feature_cuda, aabb, alphaMask_aabb, alpha_thre, valid_samples_num, gridsample_blockdim);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            if (i >= (iter_times - count_times))
                time_avg += elapsedTime / count_times;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        std::cout << "GridSample3D_2d time: " << time_avg << "ms" << std::endl;
    }
    else
    {
        if(valid_samples_num == -1){
            CalTValMin_Sharedmem_cuda(rays_chunk, aabb, near_far, tval_min, tval_min_index, N_samples, &tval_min_min, simple_mode);
            valid_samples_num = N_samples - ceil(tval_min_min * N_samples);
            if(valid_samples_num > N_samples)
                valid_samples_num = N_samples;
        }

        SampleRay_WithinHull_coord_trunc_precaltmin_validsamples_cuda_opt(rays_chunk, aabb, near_far, is_train, N_samples, valid_samples_num, rays_pts, z_vals, mask_outbbox, sampleray_blockdim, sampleray_samplesperblock);
        GridSample3D_2d_bool_dev_cuda(density_plane_line_sum, rays_pts, sigma_feature_cuda,
        aabb, alphaMask_aabb,
        valid_samples_num, alpha_thre, gridsample_blockdim);
        valid_samples_num_tensor.index_put_({0}, valid_samples_num);

    }
    return valid_samples_num;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cuda", &sampleray_gridsample, "sampleray_gridsample (CUDA)");
}
