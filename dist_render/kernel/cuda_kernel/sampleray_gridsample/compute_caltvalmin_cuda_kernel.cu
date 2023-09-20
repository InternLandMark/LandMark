#include <torch/extension.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

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
extern "C" __global__ void func_CalTValMin_kernel0(float* __restrict__ T_VAL_MIN, float* __restrict__ AABB, float* __restrict__ NEAR_FAR, float* __restrict__ RAYS_CHUNK, int N_rays) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < N_rays) {
    T_VAL_MIN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
    float O_Z = (RAYS_CHUNK[(((((int)blockIdx.x) * 3072) + (((int)threadIdx.x) * 3)) + 2)] - AABB[2]);
    float D_Z = RAYS_CHUNK[(((((int)blockIdx.x) * 3072) + (((int)threadIdx.x) * 3)) + 5)];
    float FAR = ((0.000000e+00f <= D_Z) ? NEAR_FAR[1] : ((O_Z / D_Z) * -1.000000e+00f));
    T_VAL_MIN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (1.000000e+00f - (AABB[5] / ((NEAR_FAR[0] - FAR) * D_Z)));
  }
}

void CalTValMin_cuda(torch::Tensor rays_chunk, torch::Tensor aabb, torch::Tensor near_far, torch::Tensor t_val_min)
{
    const int N_rays = rays_chunk.size(0);

    int THREADDIM = 1024;

    const dim3 grid_shape((N_rays-1)/(THREADDIM)+1, 1, 1);
    const dim3 block_shape(THREADDIM, 1, 1);
    AT_DISPATCH_FLOATING_TYPES(rays_chunk.type(), "CalTValMin", ([&] {
        func_CalTValMin_kernel0<<<grid_shape, block_shape>>>(
            t_val_min.data<float>(),
            aabb.data<float>(),
            near_far.data<float>(),
            rays_chunk.data<float>(),
            N_rays
            );
        }));
}
