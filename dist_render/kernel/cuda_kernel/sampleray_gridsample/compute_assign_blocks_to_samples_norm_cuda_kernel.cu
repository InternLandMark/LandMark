#include <torch/extension.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void func_AssignBlocksToSamplesNorm_kernel0(float* __restrict__ XYZ_SAMPLED, bool* __restrict__ RAY_VALID, float* __restrict__ B_SAMPLED, bool* __restrict__ MASKS, int N_rays, int N_samples, int plane_x, int plane_y) {
    unsigned int sample_idx = blockIdx.x * N_samples + threadIdx.x;
    B_SAMPLED[sample_idx] = 0.0;
    if (RAY_VALID[sample_idx] == true)
    {
        float xyz_x = XYZ_SAMPLED[(((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3))];
        float xyz_y = XYZ_SAMPLED[((((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3)) + 1)];
        float plane_width = (2.000000e+00f / ((float)plane_x));
        float plane_height = (2.000000e+00f / ((float)plane_y));
        float temp_0 = ((xyz_x + 1.000000e+00f) / plane_width);
        float temp_1 = ((xyz_y + 1.000000e+00f) / plane_height);
        float temp_2 = ((float)((((int)temp_0) * plane_y) + ((int)temp_1)));
        float coord_min_0 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((((int)temp_2) % plane_y) >= 0)) || ((plane_y < 0) && ((((int)temp_2) % plane_y) <= 0))) ? (((int)temp_2) / plane_y) : ((((int)temp_2) / plane_y) - 1))) * plane_width));
        float coord_min_1 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((((int)temp_2) % plane_y) >= 0)) || ((plane_y < 0) && ((((int)temp_2) % plane_y) <= 0))) ? (((int)temp_2) % plane_y) : ((((int)temp_2) % plane_y) + plane_y))) * plane_height));
        float coord_max_0 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((((int)temp_2) % plane_y) >= 0)) || ((plane_y < 0) && ((((int)temp_2) % plane_y) <= 0))) ? (((int)temp_2) / plane_y) : ((((int)temp_2) / plane_y) - 1)) + 1)) * plane_width));
        float coord_max_1 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((((int)temp_2) % plane_y) >= 0)) || ((plane_y < 0) && ((((int)temp_2) % plane_y) <= 0))) ? (((int)temp_2) % plane_y) : ((((int)temp_2) % plane_y) + plane_y)) + 1)) * plane_height));
        XYZ_SAMPLED[(((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3))] = ((((xyz_x - coord_min_0) * 2.000000e+00f) / (coord_max_0 - coord_min_0)) - 1.000000e+00f);
        XYZ_SAMPLED[((((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3)) + 1)] = ((((xyz_y - coord_min_1) * 2.000000e+00f) / (coord_max_1 - coord_min_1)) - 1.000000e+00f);
        // for (int block_idx = 0; block_idx < (plane_x * plane_y); ++block_idx) {
        //     MASKS[((((block_idx * N_rays) + ((int)blockIdx.x)) * N_samples) + ((int)threadIdx.x))] = ((signed char)(((int)temp_2) == block_idx));
        // }
        B_SAMPLED[sample_idx] = (-1.000000e+00f + ((2.000000e+00f * temp_2) / (((float)(plane_x * plane_y)) - 1.000000e+00f)));
    }
}

extern "C" __global__ void func_AssignBlocksToSamplesNormInt_kernel0(float* __restrict__ XYZ_SAMPLED, bool* __restrict__ RAY_VALID, int* __restrict__ B_SAMPLED, bool* __restrict__ MASKS, int N_rays, int N_samples, int plane_x, int plane_y) {
    unsigned int sample_idx = blockIdx.x * N_samples + threadIdx.x;
    B_SAMPLED[sample_idx] = 0;
    if (RAY_VALID[sample_idx] == true)
    {
        float xyz_x = XYZ_SAMPLED[(((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3))];
        float xyz_y = XYZ_SAMPLED[((((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3)) + 1)];
        float plane_width = (2.000000e+00f / ((float)plane_x));
        float plane_height = (2.000000e+00f / ((float)plane_y));
        float temp_0 = ((xyz_x + 1.000000e+00f) / plane_width);
        float temp_1 = ((xyz_y + 1.000000e+00f) / plane_height);
        int temp_2 = (((floorf(temp_0)) * plane_y) + (floorf(temp_1)));
        // avoid int b = 28 > 24
        if( temp_2 < 0 || temp_2 >=  (plane_x * plane_y) )
        {
            // printf("temp_2 = %d > plane_x * plane_y = %d\t", temp_2, plane_x * plane_y);
            // std::cout << "temp_2 : " << temp_2 << "> plane_x * plane_y = " << plane_x * plane_y <<std::endl;
            RAY_VALID[sample_idx] = false;
            return;
        }
        float coord_min_0 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 / plane_y) : ((temp_2 / plane_y) - 1))) * plane_width));
        float coord_min_1 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 % plane_y) : ((temp_2 % plane_y) + plane_y))) * plane_height));
        float coord_max_0 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 / plane_y) : ((temp_2 / plane_y) - 1)) + 1)) * plane_width));
        float coord_max_1 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 % plane_y) : ((temp_2 % plane_y) + plane_y)) + 1)) * plane_height));
        XYZ_SAMPLED[(((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3))] = ((((xyz_x - coord_min_0) * 2.000000e+00f) / (coord_max_0 - coord_min_0)) - 1.000000e+00f);
        XYZ_SAMPLED[((((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3)) + 1)] = ((((xyz_y - coord_min_1) * 2.000000e+00f) / (coord_max_1 - coord_min_1)) - 1.000000e+00f);

        // B_SAMPLED[sample_idx] = (-1.000000e+00f + ((2.000000e+00f * temp_2) / (((float)(plane_x * plane_y)) - 1.000000e+00f)));
        B_SAMPLED[sample_idx] = temp_2;
    }
}

void AssignBlocksToSamplesNorm_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y)
{
    int N_rays = xyz_sample.size(0);
    int N_samples = xyz_sample.size(1);
    assert(N_samples < 1024);

    const dim3 grid_shape(N_rays, 1, 1);
    const dim3 block_shape(N_samples, 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz_sample.type(), "AssignBlocksToSamplesNorm", ([&]
                                                                        {func_AssignBlocksToSamplesNorm_kernel0<<<grid_shape, block_shape>>>(
                                                                            xyz_sample.data<float>(),
                                                                            ray_valid.data<bool>(),
                                                                            b_sample.data<float>(),
                                                                            masks.data<bool>(),
                                                                            N_rays,
                                                                            N_samples,
                                                                            plane_x,
                                                                            plane_y); }));
}

void AssignBlocksToSamplesNormInt_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y)
{
    int N_rays = xyz_sample.size(0);
    int N_samples = xyz_sample.size(1);
    assert(N_samples < 1024);

    const dim3 grid_shape(N_rays, 1, 1);
    const dim3 block_shape(N_samples, 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz_sample.type(), "AssignBlocksToSamplesNormInt", ([&]
                                                                        {func_AssignBlocksToSamplesNormInt_kernel0<<<grid_shape, block_shape>>>(
                                                                            xyz_sample.data<float>(),
                                                                            ray_valid.data<bool>(),
                                                                            b_sample.data<int>(),
                                                                            masks.data<bool>(),
                                                                            N_rays,
                                                                            N_samples,
                                                                            plane_x,
                                                                            plane_y); }));
}
