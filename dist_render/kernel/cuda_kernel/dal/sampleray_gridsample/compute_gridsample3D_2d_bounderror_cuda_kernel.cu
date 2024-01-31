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
extern "C" __global__ void func_GridSample3D_2d_bounderror_kernel0(float* __restrict__ OUTPUT, float* __restrict__ INPUT, float* __restrict__ GRID, int D_out, int Batchsize, int Channel, int H_out, int W_out, int D_in, int H_in, int W_in) {
  if (((((int)blockIdx.x) * blockDim.x) + ((int)threadIdx.x)) < D_out && ((int)threadIdx.y) < H_out && ((int)threadIdx.z) < W_out) {
    float ix = __fdiv_rn(__fmul_rn(((float)__fsub_rn(W_in, 1.000000)), __fadd_rn(GRID[((((((((((int)blockIdx.x) * blockDim.x) + (((int)blockIdx.z) * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) * 3) + (((int)threadIdx.z) * 3))], 1.000000e+00f)), 2);
    float cse_var_3 = __fsub_rn(ix, ((float)((int)ix)));
    bool cse_var_2 = ((((int)ix) + 1) < W_in);
    float cse_var_1 = __fsub_rn(((float)(((int)ix) + 1.000000)), ix);
    float iy = __fdiv_rn(__fmul_rn(((float)__fsub_rn(H_in, 1.000000)), __fadd_rn(GRID[(((((((((((int)blockIdx.x) * blockDim.x) + (((int)blockIdx.z) * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) * 3) + (((int)threadIdx.z) * 3)) + 1)], 1.000000e+00f)), 2);
    float cse_var_14 = __fsub_rn(iy, ((float)((int)iy)));
    float cse_var_13 = __fsub_rn(((float)(((int)iy) + 1.000000)), iy);
    bool cse_var_11 = ((((int)iy) + 1) < H_in);
    float cse_var_10 = __fmul_rn(cse_var_3, cse_var_14);
    float cse_var_9 = __fmul_rn(cse_var_1, cse_var_14);
    bool cse_var_8 = (cse_var_2 && cse_var_11);
    float cse_var_7 = __fmul_rn(cse_var_3, cse_var_13);
    float cse_var_6 = __fmul_rn(cse_var_1, cse_var_13);
    float iz = __fdiv_rn(__fmul_rn(((float)__fsub_rn(D_in, 1.000000)), __fadd_rn(GRID[(((((((((((int)blockIdx.x) * blockDim.x) + (((int)blockIdx.z) * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) * 3) + (((int)threadIdx.z) * 3)) + 2)], 1.000000e+00f)), 2);
    float cse_var_18 = __fsub_rn(iz, ((float)((int)iz)));
    bool cse_var_17 = ((((int)iz) + 1) < D_in);
    float cse_var_16 = __fsub_rn(((float)(((int)iz) + 1.000000)), iz);
    bool bne_valid = (cse_var_2 && cse_var_17);
    bool bsw_valid = (cse_var_11 && cse_var_17);
    bool bse_valid = (cse_var_8 && cse_var_17);
    float tnw = __fmul_rn(cse_var_6, cse_var_16);
    float tne = __fmul_rn(cse_var_7, cse_var_16);
    float tsw = __fmul_rn(cse_var_9, cse_var_16);
    float tse = __fmul_rn(cse_var_10, cse_var_16);
    float bnw = __fmul_rn(cse_var_6, cse_var_18);
    float bne = __fmul_rn(cse_var_7, cse_var_18);
    float bsw = __fmul_rn(cse_var_9, cse_var_18);
    float bse = __fmul_rn(cse_var_10, cse_var_18);
    float tnw_value = __fmul_rn(INPUT[((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + ((int)iz)) * H_in) + ((int)iy)) * W_in) + ((int)ix))], tnw);
    float tne_value = (cse_var_2 ? __fmul_rn(INPUT[(((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + ((int)iz)) * H_in) + ((int)iy)) * W_in) + ((int)ix)) + 1)], tne) : 0.000000e+00f);
    float tsw_value = (cse_var_11 ? __fmul_rn(INPUT[(((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + ((int)iz)) * H_in) + ((int)iy)) + 1) * W_in) + ((int)ix))], tsw) : 0.000000e+00f);
    float tse_value = (cse_var_8 ? __fmul_rn(INPUT[((((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + ((int)iz)) * H_in) + ((int)iy)) + 1) * W_in) + ((int)ix)) + 1)], tse) : 0.000000e+00f);
    float bnw_value = (cse_var_17 ? __fmul_rn(INPUT[(((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + ((int)iz)) + 1) * H_in) + ((int)iy)) * W_in) + ((int)ix))], bnw) : 0.000000e+00f);
    float bne_value = (bne_valid ? __fmul_rn(INPUT[((((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + ((int)iz)) + 1) * H_in) + ((int)iy)) * W_in) + ((int)ix)) + 1)], bne) : 0.000000e+00f);
    float bsw_value = (bsw_valid ? __fmul_rn(INPUT[((((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + ((int)iz)) + 1) * H_in) + ((int)iy)) + 1) * W_in) + ((int)ix))], bsw) : 0.000000e+00f);
    float bse_value = (bse_valid ? __fmul_rn(INPUT[(((((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + ((int)iz)) + 1) * H_in) + ((int)iy)) + 1) * W_in) + ((int)ix)) + 1)], bse) : 0.000000e+00f);
    OUTPUT[(((((((((int)blockIdx.x) * blockDim.x) + (((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) + ((int)threadIdx.z))] = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(tnw_value, tne_value), tsw_value), tse_value), bnw_value), bne_value), bsw_value), bse_value);
  }
}

void GridSample3D_2d_bounderror_cuda(torch::Tensor input, torch::Tensor grid, torch::Tensor output, int valid_samples_num, int blockdim)
{
    // Warning! not check the lower bound of grid (should ge 0)
    const int D_out = grid.size(0);
    const int H_out = valid_samples_num;
    //const int H_out = grid.size(1);
    const int W_out = 1;
    const int Batchsize = input.size(0);
    const int Channel = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    int BLOCKDIM = blockdim;
    int THREADDIM = BLOCKDIM /(H_out * W_out);

    const dim3 grid_shape((D_out-1)/(THREADDIM)+1, Channel, Batchsize);
    const dim3 block_shape(THREADDIM, H_out, W_out);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "GridSample3D_2d_bounderror", ([&] {
        func_GridSample3D_2d_bounderror_kernel0<<<grid_shape, block_shape>>>(
            output.data<float>(),
            input.data<float>(),
            grid.data<float>(),
            D_out,
            Batchsize,
            Channel,
            H_out,
            W_out,
            D_in,
            H_in,
            W_in);
        }));
}
