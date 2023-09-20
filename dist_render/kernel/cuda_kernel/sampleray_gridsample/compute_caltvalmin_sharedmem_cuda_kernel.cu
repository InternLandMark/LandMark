#include <torch/extension.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
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
extern "C" __global__ void func_CalTValMin_sharedmem_kernel0(float* __restrict__ T_VAL_MIN, float* __restrict__ AABB, float* __restrict__ NEAR_FAR, float* __restrict__ RAYS_CHUNK, int N_rays) {
    __shared__ float s_min[1024];
    unsigned int tid = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    s_min[tid] = 1e10;
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < N_rays) {
        T_VAL_MIN[((int)blockIdx.x)] = 0.000000e+00f;
        float O_Z = (RAYS_CHUNK[(((((int)blockIdx.x) * 6144) + (((int)threadIdx.x) * 6)) + 2)] - AABB[2]);
        float D_Z = RAYS_CHUNK[(((((int)blockIdx.x) * 6144) + (((int)threadIdx.x) * 6)) + 5)];
        float FAR = ((0.000000e+00f <= D_Z) ? NEAR_FAR[1] : ((O_Z / D_Z) * -1.000000e+00f));
        s_min[tid] = (1.000000e+00f - ( (AABB[5] - AABB[2]) / ((NEAR_FAR[0] - FAR) * D_Z)));  // fix 06015
        __syncthreads();
        unsigned int n = 1024;
        while(n > 0)
        {
        if (tid < (unsigned int)(n/2))
        {
            s_min[tid] = min(s_min[tid], s_min[tid+(unsigned int)(n/2)]);
        }
        n = (unsigned int)(n/2);
        __syncthreads();
        }
        __syncthreads();
        if (tid == 0)
        T_VAL_MIN[((int)blockIdx.x)] = s_min[0];
    }
}

void CalTValMin_Sharedmem_cuda(torch::Tensor rays_chunk, torch::Tensor aabb, torch::Tensor near_far, torch::Tensor t_val_min, torch::Tensor tval_min_index, int N_samples, float* tval_min_min, bool simple_mode)
{
    const int N_rays = rays_chunk.size(0);

    int THREADDIM = 1024;

    unsigned int bdimx = (N_rays-1)/(THREADDIM)+1;
    assert (bdimx < 65536);

    const dim3 grid_shape(bdimx, 1, 1);
    const dim3 block_shape(THREADDIM, 1, 1);
    AT_DISPATCH_FLOATING_TYPES(rays_chunk.type(), "CalTValMin", ([&] {
        func_CalTValMin_sharedmem_kernel0<<<grid_shape, block_shape>>>(
            t_val_min.data<float>(),
            aabb.data<float>(),
            near_far.data<float>(),
            rays_chunk.data<float>(),
            N_rays
            );
        }));
    if (simple_mode == true)
    {
        *tval_min_min = t_val_min.index({0}).item().toFloat();
    }
    else
    {
        // auto device_ptr = thrust::device_pointer_cast<float>(t_val_min.data_ptr<float>());
        // *tval_min_min = *(thrust::min_element(device_ptr, device_ptr + t_val_min.numel()));

        auto min_classes = (torch::min)(t_val_min, 0);
        *tval_min_min = std::get<0>(min_classes).item().toFloat();
        // tval_min_index.index_put_({0}, ceil(std::get<0>(min_classes).item().toFloat() * N_samples));
    }

}
