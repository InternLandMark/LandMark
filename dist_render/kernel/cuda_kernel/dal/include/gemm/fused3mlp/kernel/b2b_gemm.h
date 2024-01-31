
#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

namespace cutlass {
namespace gemm {
namespace kernel {
template <
    typename B2bMma_ ,
    typename Epilogue_ ,
    typename ThreadblockSwizzle_ ,
    bool SplitKSerial_ >
struct B2bGemm{
public:
    using B2bMma = B2bMma_ ;
    using Epilogue = Epilogue_ ;
    using ThreadblockSwizzle = ThreadblockSwizzle_ ;
    static bool const SplitKSerial = SplitKSerial_ ;
    using OutputOp0 = typename B2bMma::OutputOp0;
    using OutputOp1 = typename B2bMma::OutputOp1;
    using OutputOp2 = typename Epilogue::OutputOp;
    using FusedAddBiasEpilogue0 = typename B2bMma::FusedAddBiasEpilogue0;
    using FusedAddBiasEpilogue1 = typename B2bMma::FusedAddBiasEpilogue1;
    using WarpCount0 = typename B2bMma::WarpCount0;
    static int const kThreadCount = 32 * WarpCount0::kCount;
struct Params{
    cutlass::gemm::GemmCoord problem_size_0;
    cutlass::gemm::GemmCoord problem_size_1;
    cutlass::gemm::GemmCoord problem_size_2;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    typename B2bMma::IteratorA0::Params params_A0;
    typename B2bMma::IteratorA0::TensorRef ref_A0;
    typename B2bMma::IteratorB0::Params params_B0;
    typename B2bMma::IteratorB0::TensorRef ref_B0;
    typename FusedAddBiasEpilogue0::OutputTileIterator::Params params_C0;
    typename FusedAddBiasEpilogue0::OutputTileIterator::TensorRef ref_C0;
    typename B2bMma::IteratorB1::Params params_B1;
    typename B2bMma::IteratorB1::TensorRef ref_B1;
    typename FusedAddBiasEpilogue1::OutputTileIterator::Params params_C1;
    typename FusedAddBiasEpilogue1::OutputTileIterator::TensorRef ref_C1;
    typename B2bMma::IteratorB2::Params params_B2;
    typename B2bMma::IteratorB2::TensorRef ref_B2;
    typename Epilogue::OutputTileIterator::Params params_C2;
    typename Epilogue::OutputTileIterator::TensorRef ref_C2;
    typename Epilogue::OutputTileIterator::Params params_D2;
    typename Epilogue::OutputTileIterator::TensorRef ref_D2;
    typename OutputOp0::Params output_op_0;
    typename OutputOp1::Params output_op_1;
    typename OutputOp2::Params output_op_2;
    int batch_count;
    int gemm_k_iterations_0;

CUTLASS_HOST_DEVICE
Params() { }


CUTLASS_HOST_DEVICE
Params(
    cutlass::gemm::GemmCoord const & problem_size_0,
    cutlass::gemm::GemmCoord const & problem_size_1,
    cutlass::gemm::GemmCoord const & problem_size_2,
    cutlass::gemm::GemmCoord const & grid_tiled_shape,
    typename B2bMma::IteratorA0::TensorRef ref_A0,
    typename B2bMma::IteratorB0::TensorRef ref_B0,
    typename FusedAddBiasEpilogue0::OutputTileIterator::TensorRef ref_C0,
    typename B2bMma::IteratorB1::TensorRef ref_B1,
    typename FusedAddBiasEpilogue1::OutputTileIterator::TensorRef ref_C1,
    typename B2bMma::IteratorB2::TensorRef ref_B2,
    typename Epilogue::OutputTileIterator::TensorRef ref_C2,
    typename Epilogue::OutputTileIterator::TensorRef ref_D2,
    typename OutputOp0::Params output_op_0 = typename OutputOp0::Params(),
    typename OutputOp1::Params output_op_1 = typename OutputOp1::Params(),
    typename OutputOp2::Params output_op_2 = typename OutputOp2::Params(),
    int batch_count = 1
):
    problem_size_0(problem_size_0),
    problem_size_1(problem_size_1),
    problem_size_2(problem_size_2),
    grid_tiled_shape(grid_tiled_shape),
    params_A0(ref_A0.layout()),
    ref_A0(ref_A0),
    params_B0(ref_B0.layout()),
    ref_B0(ref_B0),
    params_C0(ref_C0.layout()),
    ref_C0(ref_C0),
    params_B1(ref_B1.layout()),
    ref_B1(ref_B1),
    params_C1(ref_C1.layout()),
    ref_C1(ref_C1),
    params_B2(ref_B2.layout()),
    ref_B2(ref_B2),
    params_C2(ref_C2.layout()),
    ref_C2(ref_C2),
    params_D2(ref_D2.layout()),
    ref_D2(ref_D2),
    output_op_0(output_op_0),
    output_op_1(output_op_1),
    output_op_2(output_op_2),
    batch_count(batch_count) {
    gemm_k_iterations_0 = (problem_size_0.k() + B2bMma::Shape0::kK - 1) / B2bMma::Shape0::kK;
}
}; // struct Params
union SharedStorage {
    typename B2bMma::B2bMmaSharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
};
CUTLASS_HOST_DEVICE
B2bGemm() { }

CUTLASS_DEVICE
void operator()(Params const &params, SharedStorage &shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);
    int batch_idx = threadblock_tile_offset.k();
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
    params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
        return;
    }
    cutlass::MatrixCoord tb_offset_A0{
        threadblock_tile_offset.m() * B2bMma::Shape0::kM,
        0
    };
    cutlass::MatrixCoord tb_offset_B0{
        0,
        threadblock_tile_offset.n() * B2bMma::Shape0::kN
    };
    cutlass::MatrixCoord tb_offset_B1{
        0,
        threadblock_tile_offset.n() * B2bMma::Shape1::kN
    };
    cutlass::MatrixCoord tb_offset_B2{
        0,
        threadblock_tile_offset.n() * B2bMma::Shape2::kN
    };
    int thread_idx = threadIdx.x;

    MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * B2bMma::Shape2::kM,
        threadblock_tile_offset.n() * B2bMma::Shape2::kN
    );
    typename B2bMma::IteratorA0 iterator_A0(
        params.params_A0,
        params.ref_A0.data(),
        params.problem_size_0.mk(),
        thread_idx,
        tb_offset_A0);
    iterator_A0.add_pointer_offset(batch_idx * params.problem_size_0.m() * params.problem_size_0.k());

    typename B2bMma::IteratorB0 iterator_B0(
        params.params_B0,
        params.ref_B0.data(),
        params.problem_size_0.kn(),
        thread_idx,
        tb_offset_B0);
    iterator_B0.add_pointer_offset(batch_idx * params.problem_size_0.n() * params.problem_size_0.k());

    typename B2bMma::IteratorB1 iterator_B1(
        params.params_B1,
        params.ref_B1.data(),
        params.problem_size_1.kn(),
        thread_idx,
        tb_offset_B1);
    iterator_B1.add_pointer_offset(batch_idx * params.problem_size_1.n() * params.problem_size_1.k());

    typename B2bMma::IteratorB2 iterator_B2(
        params.params_B2,
        params.ref_B2.data(),
        params.problem_size_2.kn(),
        thread_idx,
        tb_offset_B2);
    iterator_B2.add_pointer_offset(batch_idx * params.problem_size_2.n() * params.problem_size_2.k());

    typename FusedAddBiasEpilogue0::OutputTileIterator iterator_C0(
        params.params_C0,
        params.ref_C0.data(),
        params.problem_size_0.mn(),
        thread_idx,
        threadblock_offset);
    int ref_C0_stride = params.ref_C0.stride()[0];
    iterator_C0.add_pointer_offset(batch_idx * params.problem_size_0.n() * (ref_C0_stride == 0 ? 1 : params.problem_size_0.m()));

    typename FusedAddBiasEpilogue1::OutputTileIterator iterator_C1(
        params.params_C1,
        params.ref_C1.data(),
        params.problem_size_1.mn(),
        thread_idx,
        threadblock_offset);
    int ref_C1_stride = params.ref_C1.stride()[0];
    iterator_C1.add_pointer_offset(batch_idx * params.problem_size_1.n() * (ref_C1_stride == 0 ? 1 : params.problem_size_1.m()));

    FusedAddBiasEpilogue0 epilogue_0;
    FusedAddBiasEpilogue1 epilogue_1;
    int warp_idx = __shfl_sync(0x1f, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    OutputOp0 output_op_0(params.output_op_0);
    OutputOp1 output_op_1(params.output_op_1);
    B2bMma b2bMma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename B2bMma::FragmentC0 src_accum;
    typename B2bMma::FragmentC2 accumulators;
    src_accum.clear();
    accumulators.clear();
    b2bMma(params.gemm_k_iterations_0, accumulators, iterator_A0, iterator_B0, iterator_B1, iterator_B2, src_accum, output_op_0, output_op_1, epilogue_0, epilogue_1, iterator_C0, iterator_C1);
    OutputOp2 output_op_2(params.output_op_2);
    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);
    typename Epilogue::OutputTileIterator iterator_C2(
        params.params_C2,
        params.ref_C2.data(),
        params.problem_size_2.mn(),
        thread_idx,
        threadblock_offset
    );
    int ref_C2_stride = params.ref_C2.stride()[0];
    iterator_C2.add_pointer_offset(batch_idx * params.problem_size_2.n() * (ref_C2_stride == 0 ? 1 : params.problem_size_2.m()));

    typename Epilogue::OutputTileIterator iterator_D2(
        params.params_D2,
        params.ref_D2.data(),
        params.problem_size_2.mn(),
        thread_idx,
        threadblock_offset
    );
    iterator_D2.add_pointer_offset(batch_idx * params.problem_size_2.n() * params.problem_size_2.m());

    Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx
    );
    epilogue(output_op_2, iterator_D2, accumulators, iterator_C2);
}
}; // struct B2bGemm
} // namespace kernel
} // namespace gemm
} // namespace cutlass
