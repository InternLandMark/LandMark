// correct the Accuracy error using haoran's method, but failed yet
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

template<typename scalar_t>
static __forceinline__ __device__
scalar_t safe_downgrade_to_int_range(scalar_t x){
    // -100.0 does not have special meaning. This is just to make sure
    // it's not within_bounds_2d or within_bounds_3d, and does not cause
    // undefined behavior. See #35506.
    if (x > INT_MAX-1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
        return static_cast<scalar_t>(-100.0);
    return x;
}


template<typename scalar_t>
static __forceinline__ __device__
scalar_t compute_coordinates(scalar_t coord, int size) {
    coord = safe_downgrade_to_int_range(coord);
    return coord;
}


template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_unnormalize(scalar_t coord, int size) {
    return ((coord + 1.f) / 2) * (size - 1);
}


template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_compute_source_index(
        scalar_t coord,
        int size) {
    coord = grid_sampler_unnormalize(coord, size);
    coord = compute_coordinates(coord, size);
    return coord;
}


static __forceinline__ __device__
bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}


__device__ float GridSample3D_2d_dev(float* INPUT, unsigned long W_in, unsigned long H_in, unsigned long D_in, float ix, float iy, float iz, unsigned long channel_offset)
{

    const int ix_tnw = floor(ix);
    const int iy_tnw = floor(iy);
    const int iz_tnw = floor(iz);

    const int ix_tne = ix_tnw + 1;
    const int iy_tne = iy_tnw;
    const int iz_tne = iz_tnw;

    const int ix_tsw = ix_tnw;
    const int iy_tsw = iy_tnw + 1;
    const int iz_tsw = iz_tnw;

    const int ix_tse = ix_tnw + 1;
    const int iy_tse = iy_tnw + 1;
    const int iz_tse = iz_tnw;

    const int ix_bnw = ix_tnw;
    const int iy_bnw = iy_tnw;
    const int iz_bnw = iz_tnw + 1;

    const int ix_bne = ix_tnw + 1;
    const int iy_bne = iy_tnw;
    const int iz_bne = iz_tnw + 1;

    const int ix_bsw = ix_tnw;
    const int iy_bsw = iy_tnw + 1;
    const int iz_bsw = iz_tnw + 1;

    const int ix_bse = ix_tnw + 1;
    const int iy_bse = iy_tnw + 1;
    const int iz_bse = iz_tnw + 1;

    float tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
    float tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
    float tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
    float tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
    float bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
    float bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
    float bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
    float bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

    float out_acc = 0.0;
    if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, D_in, H_in, W_in)) {
        // out_acc += INPUT[((((((channel_offset * D_in) + iz_tnw) * H_in) + iy_tnw) * W_in) + ix_tnw)] * tnw;
        out_acc += INPUT[IDX4C(ix_tnw, iy_tnw, iz_tnw, channel_offset, W_in, H_in, D_in)] * tnw;
    }
    if (within_bounds_3d(iz_tne, iy_tne, ix_tne, D_in, H_in, W_in)) {
        // out_acc += INPUT[(((((((channel_offset * D_in) + iz_tnw) * H_in) + iy_tnw) * W_in) + ix_tnw) + 1)] * tne;
        out_acc += INPUT[IDX4C(ix_tne, iy_tne, iz_tne, channel_offset, W_in, H_in, D_in)] * tne;
    }
    if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, D_in, H_in, W_in)) {
        // out_acc += INPUT[(((((((channel_offset * D_in) + iz_tnw) * H_in) + iy_tnw) + 1) * W_in) + ix_tnw)] * tsw;
        out_acc += INPUT[IDX4C(ix_tsw, iy_tsw, iz_tsw, channel_offset, W_in, H_in, D_in)] * tsw;
    }
    if (within_bounds_3d(iz_tse, iy_tse, ix_tse, D_in, H_in, W_in)) {
        // out_acc += INPUT[((((((((channel_offset * D_in) + iz_tnw) * H_in) + iy_tnw) + 1) * W_in) + ix_tnw) + 1)] * tse;
        out_acc += INPUT[IDX4C(ix_tse, iy_tse, iz_tse, channel_offset, W_in, H_in, D_in)] * tse;
    }
    if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, D_in, H_in, W_in)) {
        // out_acc += INPUT[(((((((channel_offset * D_in) + iz_tnw) + 1) * H_in) + iy_tnw) * W_in) + ix_tnw)] * bnw;
        out_acc += INPUT[IDX4C(ix_bnw, iy_bnw, iz_bnw, channel_offset, W_in, H_in, D_in)] * bnw;
    }
    if (within_bounds_3d(iz_bne, iy_bne, ix_bne, D_in, H_in, W_in)) {
        // out_acc += INPUT[((((((((channel_offset * D_in) + iz_tnw) + 1) * H_in) + iy_tnw) * W_in) + ix_tnw) + 1)] * bne;
        out_acc += INPUT[IDX4C(ix_bne, iy_bne, iz_bne, channel_offset, W_in, H_in, D_in)] * bne;
    }
    if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, D_in, H_in, W_in)) {
        // out_acc += INPUT[((((((((channel_offset * D_in) + iz_tnw) + 1) * H_in) + iy_tnw) + 1) * W_in) + ix_tnw)] * bsw;
        out_acc += INPUT[IDX4C(ix_bsw, iy_bsw, iz_bsw, channel_offset, W_in, H_in, D_in)] * bsw;
    }
    if (within_bounds_3d(iz_bse, iy_bse, ix_bse, D_in, H_in, W_in)) {
        // out_acc += INPUT[(((((((((channel_offset * D_in) + iz_tnw) + 1) * H_in) + iy_tnw) + 1) * W_in) + ix_tnw) + 1)] * bse;
        out_acc += INPUT[IDX4C(ix_bse, iy_bse, iz_bse, channel_offset, W_in, H_in, D_in)] * bse;
    }

    return out_acc;
}

__global__ void func_GridSample3D_2d_dev_kernel0(float* __restrict__ OUTPUT, float* __restrict__ INPUT, float* __restrict__ GRID, int D_out, int Batchsize, int Channel, int H_out, int W_out, int D_in, int H_in, int W_in) {
    if (((((int)blockIdx.x) * blockDim.x) + ((int)threadIdx.x)) < D_out && ((int)threadIdx.y) < H_out) {
        const int grid_offset = ((((((((((int)blockIdx.x) * blockDim.x) + (((int)blockIdx.z) * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) * 3) + (((int)threadIdx.z) * 3));
        float x = GRID[grid_offset];
        float ix = grid_sampler_compute_source_index(x, W_in);
        float y = GRID[grid_offset + 1];
        float iy = grid_sampler_compute_source_index(y, H_in);
        float z = GRID[grid_offset + 2];
        float iz = grid_sampler_compute_source_index(z, D_in);
        unsigned long channel_offset = blockIdx.z * Channel + blockIdx.y;

        OUTPUT[(((((((((int)blockIdx.x) * blockDim.x) + (channel_offset * D_out)) + ((int)threadIdx.x)) * H_out) + ((int)threadIdx.y)) * W_out) + ((int)threadIdx.z))] = GridSample3D_2d_dev(INPUT, W_in, H_in, D_in, ix, iy, iz, channel_offset);
    }
}

void GridSample3D_2d_dev_cuda(torch::Tensor input, torch::Tensor grid, torch::Tensor output, int valid_samples_num, int blockdim)
{
    // Warning! not check the lower bound of grid (should ge 0)
    const int D_out = grid.size(0);
    const int H_out = valid_samples_num;
    const int W_out = 1;
    const int Batchsize = input.size(0);
    const int Channel = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    int BLOCKDIM = blockdim;
    int THREADDIM = BLOCKDIM /(H_out * W_out);

    const dim3 grid_shape((D_out-1)/(THREADDIM)+1, Channel, Batchsize);
    const dim3 block_shape(THREADDIM, H_out, W_out); // H_out has to be less than 64
    AT_DISPATCH_FLOATING_TYPES(input.type(), "GridSample3D_2d_dev_cuda", ([&] {
        func_GridSample3D_2d_dev_kernel0<<<grid_shape, block_shape>>>(
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
