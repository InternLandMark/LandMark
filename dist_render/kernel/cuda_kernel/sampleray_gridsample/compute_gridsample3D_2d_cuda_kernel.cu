#include <torch/extension.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

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
extern "C" __global__ void func_GridSample3D_2d_kernel0(float* __restrict__ OUTPUT, float* __restrict__ INPUT, float* __restrict__ GRID, int D_out, int Batchsize, int Channel, int H_out, int W_out, int D_in, int H_in, int W_in) {
  if (((((int)blockIdx.x) * blockDim.x) + ((int)threadIdx.x)) < D_out && ((int)threadIdx.y) < H_out && ((int)threadIdx.z) < W_out) {
    float ix = __fmul_rn(__fmul_rn(((float)(W_in - 1)), __fadd_rn(GRID[((((((((((int)blockIdx.x) * blockDim.x) + (((int)blockIdx.z) * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) * 3) + (((int)threadIdx.z) * 3))], 1.000000e+00f)), 5.000000e-01f);
    bool cse_var_2 = (0.000000e+00f <= ix);
    float iy = __fmul_rn(__fmul_rn(((float)(H_in - 1)), __fadd_rn(GRID[(((((((((((int)blockIdx.x) * blockDim.x) + (((int)blockIdx.z) * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) * 3) + (((int)threadIdx.z) * 3)) + 1)], 1.000000e+00f)), 5.000000e-01f);
    bool cse_var_5 = (0.000000e+00f <= iy);
    float iz = __fmul_rn(__fmul_rn(((float)(D_in - 1)), __fadd_rn(GRID[(((((((((((int)blockIdx.x) * blockDim.x) + (((int)blockIdx.z) * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) * 3) + (((int)threadIdx.z) * 3)) + 2)], 1.000000e+00f)), 5.000000e-01f);
    bool cse_var_8 = (0.000000e+00f <= iz);
    bool tnw_valid = ((((((0 <= (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) && ((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) < W_in)) && (0 <= (cse_var_5 ? ((int)iy) : (((int)iy) - 1)))) && ((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) < H_in)) && (0 <= (cse_var_8 ? ((int)iz) : (((int)iz) - 1)))) && ((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) < D_in));
    bool tne_valid = ((((((-1 <= (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) && (((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) + 1) < W_in)) && (0 <= (cse_var_5 ? ((int)iy) : (((int)iy) - 1)))) && ((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) < H_in)) && (0 <= (cse_var_8 ? ((int)iz) : (((int)iz) - 1)))) && ((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) < D_in));
    bool tsw_valid = ((((((0 <= (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) && ((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) < W_in)) && (-1 <= (cse_var_5 ? ((int)iy) : (((int)iy) - 1)))) && (((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) + 1) < H_in)) && (0 <= (cse_var_8 ? ((int)iz) : (((int)iz) - 1)))) && ((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) < D_in));
    bool tse_valid = ((((((-1 <= (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) && (((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) + 1) < W_in)) && (-1 <= (cse_var_5 ? ((int)iy) : (((int)iy) - 1)))) && (((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) + 1) < H_in)) && (0 <= (cse_var_8 ? ((int)iz) : (((int)iz) - 1)))) && ((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) < D_in));
    bool bnw_valid = ((((((0 <= (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) && ((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) < W_in)) && (0 <= (cse_var_5 ? ((int)iy) : (((int)iy) - 1)))) && ((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) < H_in)) && (-1 <= (cse_var_8 ? ((int)iz) : (((int)iz) - 1)))) && (((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) + 1) < D_in));
    bool bne_valid = ((((((-1 <= (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) && (((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) + 1) < W_in)) && (0 <= (cse_var_5 ? ((int)iy) : (((int)iy) - 1)))) && ((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) < H_in)) && (-1 <= (cse_var_8 ? ((int)iz) : (((int)iz) - 1)))) && (((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) + 1) < D_in));
    bool bsw_valid = ((((((0 <= (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) && ((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) < W_in)) && (-1 <= (cse_var_5 ? ((int)iy) : (((int)iy) - 1)))) && (((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) + 1) < H_in)) && (-1 <= (cse_var_8 ? ((int)iz) : (((int)iz) - 1)))) && (((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) + 1) < D_in));
    bool bse_valid = ((((((-1 <= (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) && (((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) + 1) < W_in)) && (-1 <= (cse_var_5 ? ((int)iy) : (((int)iy) - 1)))) && (((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) + 1) < H_in)) && (-1 <= (cse_var_8 ? ((int)iz) : (((int)iz) - 1)))) && (((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) + 1) < D_in));
    float tnw = __fmul_rn(__fmul_rn(__fsub_rn(((float)((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) + 1)), ix), __fsub_rn(((float)((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) + 1)), iy)), __fsub_rn(((float)((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) + 1)), iz));
    float tne = __fmul_rn(__fmul_rn(__fsub_rn(ix, ((float)(cse_var_2 ? ((int)ix) : (((int)ix) - 1)))), __fsub_rn(((float)((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) + 1)), iy)), __fsub_rn(((float)((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) + 1)), iz));
    float tsw = __fmul_rn(__fmul_rn(__fsub_rn(((float)((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) + 1)), ix), __fsub_rn(iy, ((float)(cse_var_5 ? ((int)iy) : (((int)iy) - 1))))), __fsub_rn(((float)((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) + 1)), iz));
    float tse = __fmul_rn(__fmul_rn(__fsub_rn(ix, ((float)(cse_var_2 ? ((int)ix) : (((int)ix) - 1)))), __fsub_rn(iy, ((float)(cse_var_5 ? ((int)iy) : (((int)iy) - 1))))), __fsub_rn(((float)((cse_var_8 ? ((int)iz) : (((int)iz) - 1)) + 1)), iz));
    float bnw = __fmul_rn(__fmul_rn(__fsub_rn(((float)((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) + 1)), ix), __fsub_rn(((float)((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) + 1)), iy)), __fsub_rn(iz, ((float)(cse_var_8 ? ((int)iz) : (((int)iz) - 1)))));
    float bne = __fmul_rn(__fmul_rn(__fsub_rn(ix, ((float)(cse_var_2 ? ((int)ix) : (((int)ix) - 1)))), __fsub_rn(((float)((cse_var_5 ? ((int)iy) : (((int)iy) - 1)) + 1)), iy)), __fsub_rn(iz, ((float)(cse_var_8 ? ((int)iz) : (((int)iz) - 1)))));
    float bsw = __fmul_rn(__fmul_rn(__fsub_rn(((float)((cse_var_2 ? ((int)ix) : (((int)ix) - 1)) + 1)), ix), __fsub_rn(iy, ((float)(cse_var_5 ? ((int)iy) : (((int)iy) - 1))))), __fsub_rn(iz, ((float)(cse_var_8 ? ((int)iz) : (((int)iz) - 1)))));
    float bse = (((ix - ((float)(cse_var_2 ? ((int)ix) : (((int)ix) - 1)))) * (iy - ((float)(cse_var_5 ? ((int)iy) : (((int)iy) - 1))))) * (iz - ((float)(cse_var_8 ? ((int)iz) : (((int)iz) - 1)))));
    float tnw_value = (tnw_valid ? __fmul_rn(INPUT[((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + (cse_var_8 ? ((int)iz) : (((int)iz) - 1))) * H_in) + (cse_var_5 ? ((int)iy) : (((int)iy) - 1))) * W_in) + (cse_var_2 ? ((int)ix) : (((int)ix) - 1)))], tnw) : 0.000000e+00f);
    float tne_value = (tne_valid ? __fmul_rn(INPUT[(((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + (cse_var_8 ? ((int)iz) : (((int)iz) - 1))) * H_in) + (cse_var_5 ? ((int)iy) : (((int)iy) - 1))) * W_in) + (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) + 1)], tne) : 0.000000e+00f);
    float tsw_value = (tsw_valid ? __fmul_rn(INPUT[(((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + (cse_var_8 ? ((int)iz) : (((int)iz) - 1))) * H_in) + (cse_var_5 ? ((int)iy) : (((int)iy) - 1))) + 1) * W_in) + (cse_var_2 ? ((int)ix) : (((int)ix) - 1)))], tsw) : 0.000000e+00f);
    float tse_value = (tse_valid ? __fmul_rn(INPUT[((((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + (cse_var_8 ? ((int)iz) : (((int)iz) - 1))) * H_in) + (cse_var_5 ? ((int)iy) : (((int)iy) - 1))) + 1) * W_in) + (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) + 1)], tse) : 0.000000e+00f);
    float bnw_value = (bnw_valid ? __fmul_rn(INPUT[(((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + (cse_var_8 ? ((int)iz) : (((int)iz) - 1))) + 1) * H_in) + (cse_var_5 ? ((int)iy) : (((int)iy) - 1))) * W_in) + (cse_var_2 ? ((int)ix) : (((int)ix) - 1)))], bnw) : 0.000000e+00f);
    float bne_value = (bne_valid ? __fmul_rn(INPUT[((((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + (cse_var_8 ? ((int)iz) : (((int)iz) - 1))) + 1) * H_in) + (cse_var_5 ? ((int)iy) : (((int)iy) - 1))) * W_in) + (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) + 1)], bne) : 0.000000e+00f);
    float bsw_value = (bsw_valid ? __fmul_rn(INPUT[((((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + (cse_var_8 ? ((int)iz) : (((int)iz) - 1))) + 1) * H_in) + (cse_var_5 ? ((int)iy) : (((int)iy) - 1))) + 1) * W_in) + (cse_var_2 ? ((int)ix) : (((int)ix) - 1)))], bsw) : 0.000000e+00f);
    float bse_value = (bse_valid ? __fmul_rn(INPUT[(((((((((((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_in) + (cse_var_8 ? ((int)iz) : (((int)iz) - 1))) + 1) * H_in) + (cse_var_5 ? ((int)iy) : (((int)iy) - 1))) + 1) * W_in) + (cse_var_2 ? ((int)ix) : (((int)ix) - 1))) + 1)], bse) : 0.000000e+00f);
    OUTPUT[(((((((((int)blockIdx.x) * blockDim.x) + (((((int)blockIdx.z) * Channel) + ((int)blockIdx.y)) * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) + ((int)threadIdx.z))] = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(tnw_value, tne_value), tsw_value), tse_value), bnw_value), bne_value), bsw_value), bse_value);
  }
}

void GridSample3D_2d_cuda(torch::Tensor input, torch::Tensor grid, torch::Tensor output, int valid_samples_num, int blockdim)
{
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
    int THREADDIM = BLOCKDIM/(H_out * W_out);

    const dim3 grid_shape((D_out-1)/(THREADDIM)+1, Channel, Batchsize);
    const dim3 block_shape(THREADDIM, H_out, W_out);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "GridSample3D_2d", ([&] {
        func_GridSample3D_2d_kernel0<<<grid_shape, block_shape>>>(
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
