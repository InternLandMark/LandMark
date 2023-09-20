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

extern "C" __global__ void func_SampleRayWithinHullCoordTruncPrecaltminValidsamples_kernel0_opt(bool* __restrict__ MASK_OUTBBOX, float* __restrict__ RAYS_PTS, float* __restrict__ Z_VALS, float* __restrict__ NEAR_FAR, float* __restrict__ RAYS_CHUNK, float* __restrict__ AABB, int N_rays, int N_samples, int valid_samples_num) {
    if ((int)threadIdx.x < valid_samples_num && ((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) < N_rays)
    {
        int tvals_min_index = N_samples - valid_samples_num;
        float cse_var_1 = N_samples-1;
        MASK_OUTBBOX[(((((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) * N_samples) + ((int)threadIdx.x + tvals_min_index)))] = (bool)0;
        float O_Z = __fsub_rn(RAYS_CHUNK[(((((int)blockIdx.x) * blockDim.y * 6) + (((int)threadIdx.y) * 6)) + 2)], AABB[2]);
        float D_Z = RAYS_CHUNK[(((((int)blockIdx.x) * blockDim.y * 6) + (((int)threadIdx.y) * 6)) + 5)];
        float FAR = __fmul_rn(__fdiv_rn(O_Z, D_Z), -1.000000e+00f);
        Z_VALS[(((((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) * valid_samples_num) + ((int)threadIdx.x)))] = __fadd_rn(__fmul_rn(NEAR_FAR[0], __fsub_rn(1.000000e+00f, __fdiv_rn(((float)((int)threadIdx.x + tvals_min_index)), cse_var_1))), __fmul_rn(FAR, __fdiv_rn(((float)((int)threadIdx.x + tvals_min_index)), cse_var_1)));

        RAYS_PTS[(((((((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) * valid_samples_num) * 3) + (((int)threadIdx.x) * 3))) + 0)] = __fsub_rn(__fmul_rn(__fsub_rn(__fadd_rn(RAYS_CHUNK[(((((int)blockIdx.x) * blockDim.y * 6) + (((int)threadIdx.y) * 6)) + 0)], __fmul_rn(RAYS_CHUNK[(((((int)blockIdx.x) * blockDim.y * 6) + (((int)threadIdx.y) * 6)) + 3)], Z_VALS[(((((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) * valid_samples_num) + ((int)threadIdx.x)))])), AABB[0]), __fmul_rn(__fdividef(1.000000e+00f, __fsub_rn(AABB[(0 + 3)], AABB[0])), 2.000000e+00f)), 1.000000e+00f);
        RAYS_PTS[(((((((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) * valid_samples_num) * 3) + (((int)threadIdx.x) * 3))) + 1)] = __fsub_rn(__fmul_rn(__fsub_rn(__fadd_rn(RAYS_CHUNK[(((((int)blockIdx.x) * blockDim.y * 6) + (((int)threadIdx.y) * 6)) + 1)], __fmul_rn(RAYS_CHUNK[(((((int)blockIdx.x) * blockDim.y * 6) + (((int)threadIdx.y) * 6)) + 4)], Z_VALS[(((((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) * valid_samples_num) + ((int)threadIdx.x)))])), AABB[1]), __fmul_rn(__fdividef(1.000000e+00f, __fsub_rn(AABB[(1 + 3)], AABB[1])), 2.000000e+00f)), 1.000000e+00f);
        RAYS_PTS[(((((((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) * valid_samples_num) * 3) + (((int)threadIdx.x) * 3))) + 2)] = __fsub_rn(__fmul_rn(__fsub_rn(__fadd_rn(RAYS_CHUNK[(((((int)blockIdx.x) * blockDim.y * 6) + (((int)threadIdx.y) * 6)) + 2)], __fmul_rn(RAYS_CHUNK[(((((int)blockIdx.x) * blockDim.y * 6) + (((int)threadIdx.y) * 6)) + 5)], Z_VALS[(((((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) * valid_samples_num) + ((int)threadIdx.x)))])), AABB[2]), __fmul_rn(__fdividef(1.000000e+00f, __fsub_rn(AABB[(2 + 3)], AABB[2])), 2.000000e+00f)), 1.000000e+00f);
    }
}

void SampleRay_WithinHull_coord_trunc_precaltmin_validsamples_cuda_opt(torch::Tensor rays_chunk, torch::Tensor aabb, torch::Tensor near_far, bool is_train, int N_samples, int valid_samples_num, torch::Tensor rays_pts, torch::Tensor z_vals, torch::Tensor mask_outbbox, int blockdim, int samplesperblock)
{
    const int N_rays = rays_chunk.size(0);
    int BLOCKDIM = blockdim;
    int SAMPLESPERBLOCK = samplesperblock;
    int RAYPERBLOCK = BLOCKDIM / SAMPLESPERBLOCK;

    const dim3 grid_shape((N_rays-1)/(RAYPERBLOCK)+1, 1, 1);
    const dim3 block_shape(SAMPLESPERBLOCK, RAYPERBLOCK, 1);
    AT_DISPATCH_FLOATING_TYPES(rays_chunk.type(), "SampleRayWithinHullCoordTruncPrecaltminValidsamples", ([&] {
        func_SampleRayWithinHullCoordTruncPrecaltminValidsamples_kernel0_opt<<<grid_shape, block_shape>>>(
            mask_outbbox.data<bool>(),
            rays_pts.data<float>(),
            z_vals.data<float>(),
            near_far.data<float>(),
            rays_chunk.data<float>(),
            aabb.data<float>(),
            N_rays,
            N_samples,
            valid_samples_num
        );
    }));
}
