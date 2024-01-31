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

#define IDX2C(i,j,ld)  ((j)*(ld)+(i))
#define IDX3C(i,j,k,rown,coln)  ((k)*(rown)*(coln)+(j)*(rown)+(i))
#define IDX4C(i,j,k,n,rown,coln,depthn)  ((n)*(rown)*(coln)*(depthn)+(k)*(rown)*(coln)+(j)*(rown)+(i))

__device__ float func__AssignBlocksToSamplesNormInt_kernel0_b_masks_dev(float xyz_x, float xyz_y, int plane_y, float plane_width, float plane_height) {
    float plane_x_id = ((xyz_x + 1.000000e+00f) / plane_width);
    float plane_y_id = ((xyz_y + 1.000000e+00f) / plane_height);
    return ((floorf(plane_x_id)) * plane_y) + (floorf(plane_y_id));
}

__device__ float func__AssignBlocksToSamplesNormInt_kernel0_b_relative_masks_dev(float xyz_x, float xyz_y, int neighbour_width, float plane_width, float plane_height, int corner_block_idx_x, int corner_block_idx_y) {
    float plane_x_id = ((xyz_x + 1.000000e+00f) / plane_width) - corner_block_idx_x;
    float plane_y_id = ((xyz_y + 1.000000e+00f) / plane_height) - corner_block_idx_y;

    return ((floorf(plane_x_id)) * neighbour_width) + (floorf(plane_y_id));
}

__device__ void func__AssignBlocksToSamplesNormInt_kernel0_norm_xy_dev(float* __restrict__ XYZ_SAMPLED, int N_samples, float xyz_x, float xyz_y, int plane_y, float plane_width, float plane_height, int temp_b, int rayIdx, int ptIdx) {
    float coord_min_0 = (-1.000000e+00f + ((float)(temp_b / plane_y)) * plane_width);
    float coord_min_1 = (-1.000000e+00f + ((float)(temp_b % plane_y)) * plane_height);
    float coord_max_0 = (-1.000000e+00f + ((float)(temp_b / plane_y + 1)) * plane_width);
    float coord_max_1 = (-1.000000e+00f + ((float)(temp_b % plane_y + 1)) * plane_height);
    XYZ_SAMPLED[IDX3C(0, ptIdx, rayIdx, 3, N_samples)] = ((((xyz_x - coord_min_0) * 2.000000e+00f) / (coord_max_0 - coord_min_0)) - 1.000000e+00f);
    XYZ_SAMPLED[IDX3C(1, ptIdx, rayIdx, 3, N_samples)] = ((((xyz_y - coord_min_1) * 2.000000e+00f) / (coord_max_1 - coord_min_1)) - 1.000000e+00f);
}

__device__ void func__AssignBlocksToSamplesNormInt_kernel0_dev(float* __restrict__ XYZ_SAMPLED, bool* __restrict__ RAY_VALID, int* __restrict__ B_SAMPLED, bool* __restrict__ MASKS, int N_rays, int N_samples, int plane_x, int plane_y, int ptIdx, int rayIdx, unsigned int sample_idx) {
    float xyz_x = XYZ_SAMPLED[IDX3C(0, ptIdx, rayIdx, 3, N_samples)];
    float xyz_y = XYZ_SAMPLED[IDX3C(1, ptIdx, rayIdx, 3, N_samples)];
    float plane_width = (2.000000e+00f / ((float)plane_x));
    float plane_height = (2.000000e+00f / ((float)plane_y));
    float temp_0 = ((xyz_x + 1.000000e+00f) / plane_width);
    float temp_1 = ((xyz_y + 1.000000e+00f) / plane_height);
    int temp_2 = (((floorf(temp_0)) * plane_y) + (floorf(temp_1)));

    float coord_min_0 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 / plane_y) : ((temp_2 / plane_y) - 1))) * plane_width));
    float coord_min_1 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 % plane_y) : ((temp_2 % plane_y) + plane_y))) * plane_height));
    float coord_max_0 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 / plane_y) : ((temp_2 / plane_y) - 1)) + 1)) * plane_width));
    float coord_max_1 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 % plane_y) : ((temp_2 % plane_y) + plane_y)) + 1)) * plane_height));
    XYZ_SAMPLED[IDX3C(0, ptIdx, rayIdx, 3, N_samples)] = ((((xyz_x - coord_min_0) * 2.000000e+00f) / (coord_max_0 - coord_min_0)) - 1.000000e+00f);
    XYZ_SAMPLED[IDX3C(1, ptIdx, rayIdx, 3, N_samples)] = ((((xyz_y - coord_min_1) * 2.000000e+00f) / (coord_max_1 - coord_min_1)) - 1.000000e+00f);
    for (int block_idx = 0; block_idx < (plane_x * plane_y); ++block_idx) {
        MASKS[IDX3C(ptIdx, rayIdx, block_idx, N_samples, N_rays)] = ((signed char)(temp_2 == block_idx));
    }

    // B_SAMPLED[sample_idx] = (-1.000000e+00f + ((2.000000e+00f * temp_2) / (((float)(plane_x * plane_y)) - 1.000000e+00f)));
    B_SAMPLED[sample_idx] = temp_2;
}

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
        for (int block_idx = 0; block_idx < (plane_x * plane_y); ++block_idx) {
            MASKS[((((block_idx * N_rays) + ((int)blockIdx.x)) * N_samples) + ((int)threadIdx.x))] = ((signed char)(((int)temp_2) == block_idx));
        }
        B_SAMPLED[sample_idx] = (-1.000000e+00f + ((2.000000e+00f * temp_2) / (((float)(plane_x * plane_y)) - 1.000000e+00f)));
    }
}

extern "C" __global__ void func_AssignBlocksToSamplesNormInt_kernel0(float* __restrict__ XYZ_SAMPLED, bool* __restrict__ RAY_VALID, int* __restrict__ B_SAMPLED, bool* __restrict__ MASKS, int N_rays, int N_samples, int plane_x, int plane_y) {
    int rayIdx = (((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y);
    int ptIdx = (int)threadIdx.x;
    // unsigned int sample_idx = blockIdx.x * N_samples + threadIdx.x;
    // B_SAMPLED[sample_idx] = 0;
    if (RAY_VALID[IDX2C(ptIdx, rayIdx, N_samples)] == true)
    {
        // float xyz_x = XYZ_SAMPLED[(((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3))];
        // float xyz_y = XYZ_SAMPLED[((((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3)) + 1)];
        float xyz_x = XYZ_SAMPLED[IDX3C(0, ptIdx, rayIdx, 3, N_samples)];
        float xyz_y = XYZ_SAMPLED[IDX3C(1, ptIdx, rayIdx, 3, N_samples)];
        float plane_width = (2.000000e+00f / ((float)plane_x));
        float plane_height = (2.000000e+00f / ((float)plane_y));

        // float coord_min_0 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 / plane_y) : ((temp_2 / plane_y) - 1))) * plane_width));
        // float coord_min_1 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 % plane_y) : ((temp_2 % plane_y) + plane_y))) * plane_height));
        // float coord_max_0 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 / plane_y) : ((temp_2 / plane_y) - 1)) + 1)) * plane_width));
        // float coord_max_1 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 % plane_y) : ((temp_2 % plane_y) + plane_y)) + 1)) * plane_height));
        // XYZ_SAMPLED[(((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3))] = ((((xyz_x - coord_min_0) * 2.000000e+00f) / (coord_max_0 - coord_min_0)) - 1.000000e+00f);
        // XYZ_SAMPLED[((((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3)) + 1)] = ((((xyz_y - coord_min_1) * 2.000000e+00f) / (coord_max_1 - coord_min_1)) - 1.000000e+00f);
        // for (int block_idx = 0; block_idx < (plane_x * plane_y); ++block_idx) {
        //     MASKS[((((block_idx * N_rays) + ((int)blockIdx.x)) * N_samples) + ((int)threadIdx.x))] = ((signed char)(temp_2 == block_idx));
        // }

        // // B_SAMPLED[sample_idx] = (-1.000000e+00f + ((2.000000e+00f * temp_2) / (((float)(plane_x * plane_y)) - 1.000000e+00f)));
        // B_SAMPLED[sample_idx] = temp_2;

        // func__AssignBlocksToSamplesNormInt_kernel0_dev(XYZ_SAMPLED, RAY_VALID, B_SAMPLED, MASKS, N_rays, N_samples, plane_x, plane_y, threadIdx.x, blockIdx.x, sample_idx);
        // func__AssignBlocksToSamplesNormInt_kernel0_b_masks_dev(XYZ_SAMPLED, B_SAMPLED, MASKS, N_rays, N_samples, plane_x, plane_y, threadIdx.x, blockIdx.x, sample_idx);
        // func__AssignBlocksToSamplesNormInt_kernel0_norm_xy_dev(XYZ_SAMPLED, B_SAMPLED, N_samples, plane_x, plane_y, threadIdx.x, blockIdx.x, sample_idx);

        // output: B_SAMPLED, MASKS(removed0612)
        int temp_b = func__AssignBlocksToSamplesNormInt_kernel0_b_masks_dev(xyz_x, xyz_y, plane_y, plane_width, plane_height);

        // output: XYZ_SAMPLED
        func__AssignBlocksToSamplesNormInt_kernel0_norm_xy_dev(XYZ_SAMPLED, N_samples, xyz_x, xyz_y, plane_y, plane_width, plane_height, temp_b, rayIdx, ptIdx);

        B_SAMPLED[IDX2C(ptIdx, rayIdx, N_samples)] = temp_b;

    }
}

extern "C" __global__ void func_AssignBlocksToSamplesNormInt_relative_kernel0(float* __restrict__ XYZ_SAMPLED, bool* __restrict__ RAY_VALID, int* __restrict__ B_SAMPLED, bool* __restrict__ MASKS, int N_rays, int N_samples,
int plane_x, int plane_y, int corner_block_idx_x, int corner_block_idx_y, int neighbour_width) {
    int rayIdx = (((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y);
    int ptIdx = (int)threadIdx.x;
    // unsigned int sample_idx = blockIdx.x * N_samples + threadIdx.x;
    // B_SAMPLED[sample_idx] = 0;
    if (RAY_VALID[IDX2C(ptIdx, rayIdx, N_samples)] == true)
    {
        // float xyz_x = XYZ_SAMPLED[(((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3))];
        // float xyz_y = XYZ_SAMPLED[((((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3)) + 1)];
        float xyz_x = XYZ_SAMPLED[IDX3C(0, ptIdx, rayIdx, 3, N_samples)];
        float xyz_y = XYZ_SAMPLED[IDX3C(1, ptIdx, rayIdx, 3, N_samples)];
        float plane_width = (2.000000e+00f / ((float)plane_x));
        float plane_height = (2.000000e+00f / ((float)plane_y));
        // output: B_SAMPLED, MASKS(removed0612)
        int temp_b = func__AssignBlocksToSamplesNormInt_kernel0_b_masks_dev(xyz_x, xyz_y, plane_y, plane_width, plane_height);

        // output: XYZ_SAMPLED
        func__AssignBlocksToSamplesNormInt_kernel0_norm_xy_dev(XYZ_SAMPLED, N_samples, xyz_x, xyz_y, plane_y, plane_width, plane_height, temp_b, rayIdx, ptIdx);

        temp_b = func__AssignBlocksToSamplesNormInt_kernel0_b_relative_masks_dev(xyz_x, xyz_y, neighbour_width, plane_width, plane_height, corner_block_idx_x, corner_block_idx_y);
        B_SAMPLED[IDX2C(ptIdx, rayIdx, N_samples)] = temp_b;
    }
}

extern "C" __global__ void func_AssignBlocksToSamplesNormInt_fused_relative_kernel0(float* __restrict__ XYZ_SAMPLED, bool* __restrict__ RAY_VALID, int* __restrict__ B_SAMPLED,  int N_rays, int N_samples,
int plane_x, int plane_y, int corner_block_idx_x, int corner_block_idx_y, int neighbour_width,
float** plane_line_ptr_not_share, float* sigma, const int* hw_in_not_share, const int block_in, const int C, const int arraySize, bool judgeOverflow) {

    // 给参数数组开一个sharedmemory空间, 其中10为arraySize能取到的最大值.
    __shared__ int hw_in[10 * 4];
    __shared__ float* plane_line_ptr[10 * 2];
    int tid = (int)threadIdx.x;
    if (tid < arraySize * 4){
        hw_in[tid] = hw_in_not_share[tid];
    }else if(tid < arraySize * 4 + arraySize * 2){
        plane_line_ptr[tid - arraySize * 4] = plane_line_ptr_not_share[tid - arraySize * 4];
    }
    __syncthreads();

    int rayIdx = (((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y);
    int ptIdx = (int)threadIdx.x;
    // unsigned int sample_idx = blockIdx.x * N_samples + threadIdx.x;
    // B_SAMPLED[sample_idx] = 0;
    if (RAY_VALID[IDX2C(ptIdx, rayIdx, N_samples)] == true)
    {
        // float xyz_x = XYZ_SAMPLED[(((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3))];
        // float xyz_y = XYZ_SAMPLED[((((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3)) + 1)];
        float xyz_x = XYZ_SAMPLED[IDX3C(0, ptIdx, rayIdx, 3, N_samples)];
        float xyz_y = XYZ_SAMPLED[IDX3C(1, ptIdx, rayIdx, 3, N_samples)];
        float plane_width = (2.000000e+00f / ((float)plane_x));
        float plane_height = (2.000000e+00f / ((float)plane_y));

        // float coord_min_0 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 / plane_y) : ((temp_2 / plane_y) - 1))) * plane_width));
        // float coord_min_1 = (-1.000000e+00f + (((float)((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 % plane_y) : ((temp_2 % plane_y) + plane_y))) * plane_height));
        // float coord_max_0 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 / plane_y) : ((temp_2 / plane_y) - 1)) + 1)) * plane_width));
        // float coord_max_1 = (-1.000000e+00f + (((float)(((((plane_y >= 0) && ((temp_2 % plane_y) >= 0)) || ((plane_y < 0) && ((temp_2 % plane_y) <= 0))) ? (temp_2 % plane_y) : ((temp_2 % plane_y) + plane_y)) + 1)) * plane_height));
        // XYZ_SAMPLED[(((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3))] = ((((xyz_x - coord_min_0) * 2.000000e+00f) / (coord_max_0 - coord_min_0)) - 1.000000e+00f);
        // XYZ_SAMPLED[((((((int)blockIdx.x) * N_samples) * 3) + (((int)threadIdx.x) * 3)) + 1)] = ((((xyz_y - coord_min_1) * 2.000000e+00f) / (coord_max_1 - coord_min_1)) - 1.000000e+00f);
        // for (int block_idx = 0; block_idx < (plane_x * plane_y); ++block_idx) {
        //     MASKS[((((block_idx * N_rays) + ((int)blockIdx.x)) * N_samples) + ((int)threadIdx.x))] = ((signed char)(temp_2 == block_idx));
        // }

        // // B_SAMPLED[sample_idx] = (-1.000000e+00f + ((2.000000e+00f * temp_2) / (((float)(plane_x * plane_y)) - 1.000000e+00f)));
        // B_SAMPLED[sample_idx] = temp_2;

        // func__AssignBlocksToSamplesNormInt_kernel0_dev(XYZ_SAMPLED, RAY_VALID, B_SAMPLED, MASKS, N_rays, N_samples, plane_x, plane_y, threadIdx.x, blockIdx.x, sample_idx);
        // func__AssignBlocksToSamplesNormInt_kernel0_b_masks_dev(XYZ_SAMPLED, B_SAMPLED, MASKS, N_rays, N_samples, plane_x, plane_y, threadIdx.x, blockIdx.x, sample_idx);
        // func__AssignBlocksToSamplesNormInt_kernel0_norm_xy_dev(XYZ_SAMPLED, B_SAMPLED, N_samples, plane_x, plane_y, threadIdx.x, blockIdx.x, sample_idx);

        // output: B_SAMPLED, MASKS(removed0612)
        int temp_b = func__AssignBlocksToSamplesNormInt_kernel0_b_masks_dev(xyz_x, xyz_y, plane_y, plane_width, plane_height);

        // output: XYZ_SAMPLED
        func__AssignBlocksToSamplesNormInt_kernel0_norm_xy_dev(XYZ_SAMPLED, N_samples, xyz_x, xyz_y, plane_y, plane_width, plane_height, temp_b, rayIdx, ptIdx);

        temp_b = func__AssignBlocksToSamplesNormInt_kernel0_b_relative_masks_dev(xyz_x, xyz_y, neighbour_width, plane_width, plane_height, corner_block_idx_x, corner_block_idx_y);
        B_SAMPLED[IDX2C(ptIdx, rayIdx, N_samples)] = temp_b;




        // fused part
        float sum;
        sum = 0.0;
        unsigned int xyzlen = 3; //4
        int ib = temp_b;
        float ix, iy, iz;
        float result_plane, result_line;
        // int m_id = (((int)blockIdx.x * num_block) + (int)threadIdx.x);

        // 遍历每个分辨率
        for (int planeid = 0; planeid < arraySize; planeid++){
        // hw_in按照h_in, w_in, line_h_in, line_w_in...存放
        // 按典型数值:plane:1x16x24x493x920, line: 1x16x310x24, 对应下放四个数值为:
                        // IDX2C(param_id, plane_id, param_len), 其中param_len =  4
        const int h_in = hw_in[planeid * 4 + 0];      // 493
        const int w_in = hw_in[planeid * 4 + 1];      // 920
        const int line_h_in = hw_in[planeid * 4 + 2]; // 310
        const int line_w_in = hw_in[planeid * 4 + 3]; // 24
        // plane_line_ptr按照plane_ptr, line_ptr...存放
                                            // IDX2C(ptr_id, plane_id, ptr_len), 其中ptr_len = 2
        const float * plane = plane_line_ptr[planeid * 2 + 0];
        const float * line = plane_line_ptr[planeid * 2 + 1];

                                // IDX2C(xyz_id, m_pos, xyzlen), 其中m_pos = (((int)blockIdx.x * num_block) + (int)threadIdx.x)
        ix = ((float(w_in-1) * (XYZ_SAMPLED[IDX3C(0, ptIdx, rayIdx, 3, N_samples)] + 1.0)) / 2.0);
        iy = ((float(h_in-1) * (XYZ_SAMPLED[IDX3C(1, ptIdx, rayIdx, 3, N_samples)] + 1.0)) / 2.0);
        long b_pos = long(ib) * h_in * w_in;
        int ix_nw = (int)floorf(ix);
        int iy_nw = (int)floorf(iy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;
        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);

        iz = ((float(line_h_in-1) * (XYZ_SAMPLED[IDX3C(2, ptIdx, rayIdx, 3, N_samples)] + 1.0)) / 2.0);
        int32_t iz_nw = int32_t(iz);

        // 遍历每个channel
        for (int cc = 0; cc < C; cc++) {
            if (judgeOverflow && (abs(XYZ_SAMPLED[IDX3C(0, ptIdx, rayIdx, 3, N_samples)]) > 1.0 || abs(XYZ_SAMPLED[IDX3C(1, ptIdx, rayIdx, 3, N_samples)]) > 1.0)){
            result_plane = 0.0;
            }else{
            long channel_pos = long(cc) * block_in * h_in * w_in;
            float nw_val  = ((ix_nw >= 0) && (iy_nw >= 0) && (ix_nw < w_in) && (iy_nw < h_in)) ?
                // IDX4C(ix_nw, iy_nw, ib ,cc, w_in, h_in, block_in)
                plane[channel_pos + b_pos + iy_nw * w_in + ix_nw ] : 0;
            float ne_val  = ((ix_ne >= 0) && (iy_ne >= 0) && (ix_ne < w_in) && (iy_ne < h_in)) ?
                // IDX4C(min((ix_nw + 1), (w_in-1)), iy_nw, ib ,cc, w_in, h_in, block_in)
                plane[channel_pos + b_pos + iy_nw * w_in + min((ix_nw + 1), (w_in-1))] : 0;
            float sw_val  = ((ix_sw >= 0) && (iy_sw >= 0) && (ix_sw < w_in) && (iy_sw < h_in)) ?
                // IDX4C(ix_nw, min((iy_nw + 1), (h_in-1)), ib ,cc, w_in, h_in, block_in)
                plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + ix_nw] : 0;
            float se_val  = ((ix_se >= 0) && (iy_se >= 0) && (ix_se < w_in) && (iy_se < h_in)) ?
                // IDX4C(min((ix_nw + 1), (w_in-1)), min((iy_nw + 1), (h_in-1)), ib ,cc, w_in, h_in, block_in)
                plane[channel_pos + b_pos + min((iy_nw + 1), (h_in-1)) * w_in + min((ix_nw + 1), (w_in-1))] : 0;
            result_plane = nw_val * nw +  ne_val * ne + sw_val * sw +  se_val * se;
            }

            if (judgeOverflow && (abs(XYZ_SAMPLED[IDX3C(2, ptIdx, rayIdx, 3, N_samples)]) > 1.0)){
            result_line = 0.0;
            }else{                // IDX3C(ib, iz_nw, cc, line_w_in, line_h_in)
            result_line = ((line[cc * line_h_in * line_w_in + iz_nw * line_w_in + ib] * (float(iz_nw + 1) - iz))) +
                ((line[cc * line_h_in * line_w_in + min((iz_nw + 1), (line_h_in-1)) * line_w_in + ib] * (iz - float(iz_nw))));
            }
            sum += result_plane * result_line;
        }
        }
        sigma[IDX2C(ptIdx, rayIdx, N_samples)] = sum;
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

void AssignBlocksToSamplesNormInt_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks, int plane_x, int plane_y, int blockdim, int samplesperblock)
{
    int N_rays = xyz_sample.size(0);
    // int N_samples = xyz_sample.size(1);
    // assert(N_samples < 1024);
    int BLOCKDIM = blockdim;
    int SAMPLESPERBLOCK = xyz_sample.size(1);
    int RAYPERBLOCK = BLOCKDIM / SAMPLESPERBLOCK;

    // const dim3 grid_shape(N_rays, 1, 1);
    // const dim3 block_shape(N_samples, 1, 1);
    const dim3 grid_shape((N_rays-1)/(RAYPERBLOCK)+1, 1, 1);
    const dim3 block_shape(SAMPLESPERBLOCK, RAYPERBLOCK, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz_sample.type(), "AssignBlocksToSamplesNormInt", ([&]
                                                                        {func_AssignBlocksToSamplesNormInt_kernel0<<<grid_shape, block_shape>>>(
                                                                            xyz_sample.data<float>(),
                                                                            ray_valid.data<bool>(),
                                                                            b_sample.data<int>(),
                                                                            masks.data<bool>(),
                                                                            N_rays,
                                                                            SAMPLESPERBLOCK,
                                                                            plane_x,
                                                                            plane_y); }));
}
void AssignBlocksToSamplesNormInt_relative_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample, torch::Tensor masks,
int plane_x, int plane_y, int corner_block_idx_x, int corner_block_idx_y, int neighbour_width, int blockdim, int samplesperblock)
{
    int N_rays = xyz_sample.size(0);
    // int N_samples = xyz_sample.size(1);
    // assert(N_samples < 1024);
    int BLOCKDIM = blockdim;
    int SAMPLESPERBLOCK = xyz_sample.size(1);
    int RAYPERBLOCK = BLOCKDIM / SAMPLESPERBLOCK;

    // const dim3 grid_shape(N_rays, 1, 1);
    // const dim3 block_shape(N_samples, 1, 1);
    const dim3 grid_shape((N_rays-1)/(RAYPERBLOCK)+1, 1, 1);
    const dim3 block_shape(SAMPLESPERBLOCK, RAYPERBLOCK, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz_sample.type(), "AssignBlocksToSamplesNormInt", ([&]
                                                                        {func_AssignBlocksToSamplesNormInt_relative_kernel0<<<grid_shape, block_shape>>>(
                                                                            xyz_sample.data<float>(),
                                                                            ray_valid.data<bool>(),
                                                                            b_sample.data<int>(),
                                                                            masks.data<bool>(),
                                                                            N_rays,
                                                                            SAMPLESPERBLOCK,
                                                                            plane_x,
                                                                            plane_y,
                                                                            corner_block_idx_x, corner_block_idx_y, neighbour_width); }));
}

void AssignBlocksToSamplesNormInt_fused_relative_cuda(torch::Tensor xyz_sample, torch::Tensor ray_valid, torch::Tensor b_sample,
int plane_x, int plane_y, int corner_block_idx_x, int corner_block_idx_y, int neighbour_width, int blockdim, int samplesperblock,
std::vector<torch::Tensor> plane, std::vector<torch::Tensor> line,
torch::Tensor hw_in, torch::Tensor plane_line_ptr, torch::Tensor sigma)
{
    int N_rays = xyz_sample.size(0);
    // int N_samples = xyz_sample.size(1);
    // assert(N_samples < 1024);
    int BLOCKDIM = blockdim;
    int SAMPLESPERBLOCK = xyz_sample.size(1);
    int RAYPERBLOCK = BLOCKDIM / SAMPLESPERBLOCK;

    // const dim3 grid_shape(N_rays, 1, 1);
    // const dim3 block_shape(N_samples, 1, 1);

    // fused part
    // plane shape (N,C,h_in,w_in)
    const int n = plane[0].size(0); // n == 1
    const int c = plane[0].size(1); // c == 16
    const int block_in = plane[0].size(2); // block_in == 8

    // plane_line_ptr的赋值在此进行:
    int arraySize = plane.size();    // tensorf.density_plane的长度,只测试过为3的情况, 待补充为2情况的测试
    float* plane_line_ptr_cpu[arraySize * 2];       // 各个plane和line平面数据的指针
    // 赋值参数数组, 在cpu端
    for(int idx = 0; idx < plane.size(); idx++){
      plane_line_ptr_cpu[idx * 2] = plane[idx].data<float>();
      plane_line_ptr_cpu[idx * 2 + 1] = line[idx].data<float>();
    }
    // 拷贝到GPU端
    float** plane_line_ptr_gpu = (float**)plane_line_ptr.data<int>();
    cudaMemcpy(plane_line_ptr_gpu, plane_line_ptr_cpu, arraySize * sizeof(float*) * 2, cudaMemcpyHostToDevice);



    const dim3 grid_shape((N_rays-1)/(RAYPERBLOCK)+1, 1, 1);
    const dim3 block_shape(SAMPLESPERBLOCK, RAYPERBLOCK, 1);
    AT_DISPATCH_FLOATING_TYPES(xyz_sample.type(), "AssignBlocksToSamplesNormInt", ([&]
                                                                        {func_AssignBlocksToSamplesNormInt_fused_relative_kernel0<<<grid_shape, block_shape>>>(
                                                                            xyz_sample.data<float>(),
                                                                            ray_valid.data<bool>(),
                                                                            b_sample.data<int>(),

                                                                            N_rays,
                                                                            SAMPLESPERBLOCK,
                                                                            plane_x,
                                                                            plane_y,
                                                                            corner_block_idx_x, corner_block_idx_y, neighbour_width,
                                                                            plane_line_ptr_gpu,
                                                                            sigma.data<float>(),
                                                                            hw_in.data<int>(),
                                                                            block_in, c, arraySize, false
                                                                            ); }));

}
