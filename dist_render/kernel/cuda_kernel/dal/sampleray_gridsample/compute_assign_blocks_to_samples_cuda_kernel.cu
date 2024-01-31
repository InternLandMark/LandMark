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
extern "C" __global__ void __launch_bounds__(1024) func_AssignBlocksToSamples_kernel0(bool *__restrict__ MASKS, float *__restrict__ XYZ_SAMPLED_VALID, float *__restrict__ OUTPUT, int valid_num, int plane_x, int plane_y)
{
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < valid_num)
    {
        float cse_var_2 = (2.000000e+00f / ((float)plane_y));
        float cse_var_1 = (2.000000e+00f / ((float)plane_x));
        MASKS[(((((int)blockIdx.x) * 1024) + (((((int)blockIdx.z) * plane_y) + ((int)blockIdx.y)) * valid_num)) + ((int)threadIdx.x))] = (bool)1;
        MASKS[(((((int)blockIdx.x) * 1024) + (((((int)blockIdx.z) * plane_y) + ((int)blockIdx.y)) * valid_num)) + ((int)threadIdx.x))] = ((bool)(((((-1.000000e+00f + (cse_var_1 * ((float)((int)blockIdx.z)))) <= XYZ_SAMPLED_VALID[((((int)blockIdx.x) * 3072) + (((int)threadIdx.x) * 3))]) && ((((int)blockIdx.z) == (plane_x - 1)) ? (XYZ_SAMPLED_VALID[((((int)blockIdx.x) * 3072) + (((int)threadIdx.x) * 3))] <= (-1.000000e+00f + (cse_var_1 * ((float)(((int)blockIdx.z) + 1))))) : (XYZ_SAMPLED_VALID[((((int)blockIdx.x) * 3072) + (((int)threadIdx.x) * 3))] < (-1.000000e+00f + (cse_var_1 * ((float)(((int)blockIdx.z) + 1))))))) && ((-1.000000e+00f + (cse_var_2 * ((float)((int)blockIdx.y)))) <= XYZ_SAMPLED_VALID[(((((int)blockIdx.x) * 3072) + (((int)threadIdx.x) * 3)) + 1)])) && ((((int)blockIdx.y) == (plane_y - 1)) ? (XYZ_SAMPLED_VALID[(((((int)blockIdx.x) * 3072) + (((int)threadIdx.x) * 3)) + 1)] <= (-1.000000e+00f + (cse_var_2 * ((float)(((int)blockIdx.y) + 1))))) : (XYZ_SAMPLED_VALID[(((((int)blockIdx.x) * 3072) + (((int)threadIdx.x) * 3)) + 1)] < (-1.000000e+00f + (cse_var_2 * ((float)(((int)blockIdx.y) + 1))))))));
        if (((bool)MASKS[(((((int)blockIdx.x) * 1024) + (((((int)blockIdx.z) * plane_y) + ((int)blockIdx.y)) * valid_num)) + ((int)threadIdx.x))]) == (bool)1)
        {
            OUTPUT[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (-1.000000e+00f + ((2.000000e+00f * ((float)((((int)blockIdx.z) * plane_y) + ((int)blockIdx.y)))) / ((float)((plane_x * plane_y) - 1))));
        }
    }
}

void AssignBlocksToSamples_cuda(torch::Tensor masks, torch::Tensor xyz_sample_valid, torch::Tensor output, int valid_num, int plane_x, int plane_y, int blockdim)
{

    int BLOCKDIM = blockdim;

    unsigned int bdimx = (valid_num - 1) / (BLOCKDIM) + 1;
    // assert(bdimx < 65536);

    const dim3 grid_shape(bdimx, plane_y, plane_x);
    const dim3 block_shape(BLOCKDIM, 1, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz_sample_valid.type(), "AssignBlocksToSamples", ([&]
                                                                       { func_AssignBlocksToSamples_kernel0<<<grid_shape, block_shape>>>(
                                                                             masks.data<bool>(),
                                                                             xyz_sample_valid.data<float>(),
                                                                             output.data<float>(),
                                                                             valid_num,
                                                                             plane_x,
                                                                             plane_y); }));
}
