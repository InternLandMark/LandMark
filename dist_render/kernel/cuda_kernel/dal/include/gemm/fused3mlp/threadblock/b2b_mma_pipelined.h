
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"

#include "gemm/fused3mlp/threadblock/b2b_mma_base.h"
namespace cutlass {
namespace gemm {
namespace threadblock {
template <
    typename Shape0_ ,
    typename IteratorA0_ ,
    typename SmemIteratorA0_ ,
    typename IteratorB0_ ,
    typename SmemIteratorB0_ ,
    typename Shape1_ ,
    typename FragmentIteratorA1_ ,
    typename IteratorB1_ ,
    typename SmemIteratorB1_ ,
    typename Shape2_ ,
    typename FragmentIteratorA2_ ,
    typename IteratorB2_ ,
    typename SmemIteratorB2_ ,
    typename ElementC_ ,
    typename LayoutC_ ,
    typename OutputOp0_ ,
    typename OutputOp1_ ,
    typename FusedAddBiasEpilogue0_ ,
    typename FusedAddBiasEpilogue1_ ,
    typename Policy0_ ,
    typename Policy1_ ,
    typename Policy2_ ,
    int Stage0_ ,
    int Stage1_ ,
    int Stage2_ ,
    typename TransformA0_ = NumericArrayConverter<typename SmemIteratorA0_::Element, typename IteratorA0_::Element, IteratorA0_::Fragment::kElements> ,
    typename TransformB0_ = NumericArrayConverter<typename SmemIteratorB0_::Element, typename IteratorB0_::Element, IteratorB0_::Fragment::kElements> ,
    typename TransformB1_ = NumericArrayConverter<typename SmemIteratorB1_::Element, typename IteratorB1_::Element, IteratorB1_::Fragment::kElements> ,
    typename TransformB2_ = NumericArrayConverter<typename SmemIteratorB2_::Element, typename IteratorB2_::Element, IteratorB2_::Fragment::kElements> ,
    typename Enable_ = bool >
class B2bMmaPipelined : public B2bMmaBase<Shape0_, Shape1_, Shape2_, Policy0_, Policy1_, Policy2_, Stage0_, Stage1_, Stage2_>{
public:
    using Shape0 = Shape0_ ;
    using IteratorA0 = IteratorA0_ ;
    using SmemIteratorA0 = SmemIteratorA0_ ;
    using IteratorB0 = IteratorB0_ ;
    using SmemIteratorB0 = SmemIteratorB0_ ;
    using Shape1 = Shape1_ ;
    using FragmentIteratorA1 = FragmentIteratorA1_ ;
    using IteratorB1 = IteratorB1_ ;
    using SmemIteratorB1 = SmemIteratorB1_ ;
    using Shape2 = Shape2_ ;
    using FragmentIteratorA2 = FragmentIteratorA2_ ;
    using IteratorB2 = IteratorB2_ ;
    using SmemIteratorB2 = SmemIteratorB2_ ;
    using ElementC = ElementC_ ;
    using LayoutC = LayoutC_ ;
    using OutputOp0 = OutputOp0_ ;
    using OutputOp1 = OutputOp1_ ;
    using FusedAddBiasEpilogue0 = FusedAddBiasEpilogue0_ ;
    using FusedAddBiasEpilogue1 = FusedAddBiasEpilogue1_ ;
    using Policy0 = Policy0_ ;
    using Policy1 = Policy1_ ;
    using Policy2 = Policy2_ ;
    static int const Stage0 = Stage0_ ;
    static int const Stage1 = Stage1_ ;
    static int const Stage2 = Stage2_ ;
    using TransformA0 = TransformA0_ ;
    using TransformB0 = TransformB0_ ;
    using TransformB1 = TransformB1_ ;
    using TransformB2 = TransformB2_ ;
    using Enable = Enable_ ;
using FragmentA0 = typename IteratorA0::Fragment;
using Base = B2bMmaBase<Shape0_, Shape1_, Shape2_, Policy0_, Policy1_, Policy2_, Stage0_, Stage1_, Stage2_>;
using FragmentB0 = typename IteratorB0::Fragment;
using FragmentC0 = typename Policy0::Operator::FragmentC;
using Operator0 = typename Policy0::Operator;
using FragmentB1 = typename IteratorB1::Fragment;
using FragmentC1 = typename Policy1::Operator::FragmentC;
using Operator1 = typename Policy1::Operator;
using FragmentB2 = typename IteratorB2::Fragment;
using FragmentC2 = typename Policy2::Operator::FragmentC;
using Operator2 = typename Policy2::Operator;
using IteratorC0 = typename FusedAddBiasEpilogue0::OutputTileIterator;
using IteratorC1 = typename FusedAddBiasEpilogue1::OutputTileIterator;
using ArchTag = typename Policy0::Operator::ArchTag;
static ComplexTransform const kTransformA0 = Operator0::kTransformA;
static ComplexTransform const kTransformB0 = Operator0::kTransformB;
static ComplexTransform const kTransformB1 = Operator1::kTransformB;
static ComplexTransform const kTransformB2 = Operator2::kTransformB;
private:
using WarpFragmentA0 = typename Operator0::FragmentA;
using WarpFragmentB0 = typename Operator0::FragmentB;
using WarpFragmentA1 = typename FragmentIteratorA1::Fragment;
using WarpFragmentB1 = typename Operator1::FragmentB;
using WarpFragmentA2 = typename FragmentIteratorA2::Fragment;
using WarpFragmentB2 = typename Operator2::FragmentB;
protected:
SmemIteratorA0 smem_iterator_A_;
SmemIteratorB0 smem_iterator_B0_;
SmemIteratorB1 smem_iterator_B1_;
SmemIteratorB2 smem_iterator_B2_;
public:
 CUTLASS_DEVICE
  void operator()(
 int gemm_k_iterations_0,
FragmentC2 &accum2,
IteratorA0 iterator_A,
IteratorB0 iterator_B0,
IteratorB1 iterator_B1,
IteratorB2 iterator_B2,
FragmentC0 const &src_accum,
OutputOp0 output_op_0,
OutputOp1 output_op_1,
FusedAddBiasEpilogue0 epilogue_0,
FusedAddBiasEpilogue1 epilogue_1,
IteratorC0 iterator_C0,
IteratorC1 iterator_C1,
TransformA0 transform_A0 = TransformA0(),
TransformB0 transform_B0 = TransformB0(),
TransformB1 transform_B1 = TransformB1(),
TransformB2 transform_B2 = TransformB2()
) {
     FragmentC0 accum0 = src_accum;

    FragmentA0 tb_frag_A;
    FragmentB0 tb_frag_B0;

    tb_frag_A.clear();
    tb_frag_B0.clear();

    // The last kblock is loaded in the prolog
    iterator_A.load(tb_frag_A);
    iterator_B0.load(tb_frag_B0);

    ++iterator_A;
    ++iterator_B0;

    this->smem_iterator_A_.store(tb_frag_A);
    this->smem_iterator_B0_.store(tb_frag_B0);

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B0_;

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA0 warp_frag_A0[2];
    WarpFragmentB0 warp_frag_B0[2];

    this->warp_tile_iterator_A0_.set_kgroup_index(0);
    this->warp_tile_iterator_B0_.set_kgroup_index(0);

    this->warp_tile_iterator_A0_.load(warp_frag_A0[0]);
    this->warp_tile_iterator_B0_.load(warp_frag_B0[0]);

    ++this->warp_tile_iterator_A0_;
    ++this->warp_tile_iterator_B0_;

    Operator0 warp_mma0;

    int smem_write_stage_idx = 1;

    // Avoid reading out of bounds
    if (gemm_k_iterations_0 <= 1) {
      iterator_A.clear_mask();
      iterator_B0.clear_mask();
    }

    // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing
    // shared memory loads (which have the tightest latency requirement).
    iterator_A.load(tb_frag_A);

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::WarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations_0 > 0; --gemm_k_iterations_0) {

      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations0; ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations0 - 1) {

          // Write fragments to shared memory
          this->smem_iterator_A_.store(tb_frag_A);

          this->smem_iterator_B0_.store(tb_frag_B0);

          __syncthreads();

          // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing
          // shared memory loads (which have the tightest latency requirement).
          iterator_A.load(tb_frag_A);

          ++this->smem_iterator_B0_;
          ++this->smem_iterator_A_;


          // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::Stage0});
            this->smem_iterator_B0_.add_tile_offset({-Base::Stage0, 0});
          }
          else {
            this->warp_tile_iterator_A0_.add_tile_offset(
                {0, -Base::Stage0 * Policy0::kPartitionsK * Base::kWarpGemmIterations0});
            this->warp_tile_iterator_B0_.add_tile_offset(
                {-Base::Stage0 * Policy0::kPartitionsK * Base::kWarpGemmIterations0,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A0_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations0);
        this->warp_tile_iterator_B0_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations0);

        this->warp_tile_iterator_A0_.load(warp_frag_A0[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B0_.load(warp_frag_B0[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A0_;
        ++this->warp_tile_iterator_B0_;

        if (warp_mma_k == 0) {

          iterator_B0.load(tb_frag_B0);

          ++iterator_A;
          ++iterator_B0;

          // Avoid reading out of bounds if this was the last loop iteration
          if (gemm_k_iterations_0 <= 2) {
            iterator_A.clear_mask();
            iterator_B0.clear_mask();
          }
        }

        warp_mma0(accum0, warp_frag_A0[warp_mma_k % 2], warp_frag_B0[warp_mma_k % 2], accum0);
      }
    }
    FragmentC1 accum1;
    accum1.clear();
// 2 Gemm    /// Iterator to load a warp-scoped tile of A1 operand from intermediate accumulator tile
    FragmentC0 after_epilogue_accu0;
    epilogue_0(output_op_0, accum0, after_epilogue_accu0, iterator_C0);
    FragmentIteratorA1 warp_tile_iterator_A1_(after_epilogue_accu0);
    FragmentB1 tb_frag_B1;
    tb_frag_B1.clear();
    iterator_B1.load(tb_frag_B1);
    ++iterator_B1;
    this->smem_iterator_B1_.store(tb_frag_B1);
    ++this->smem_iterator_B1_;
    __syncthreads();
    WarpFragmentA1 warp_frag_A1[2];
    WarpFragmentB1 warp_frag_B1[2];
    this->warp_tile_iterator_B1_.set_kgroup_index(0);
    warp_tile_iterator_A1_.load(warp_frag_A1[0]);
    this->warp_tile_iterator_B1_.load(warp_frag_B1[0]);
    ++warp_tile_iterator_A1_;
    ++this->warp_tile_iterator_B1_;
    Operator1 warp_mma1;
    smem_write_stage_idx = 1;
    int gemm_k_iterations_1 = FragmentIteratorA1::Policy::kIterations / Base::kWarpGemmIterations1;
    if (gemm_k_iterations_1 <= 1 ){
        iterator_B1.clear_mask();
    }
    CUTLASS_PRAGMA_UNROLL
    for (; gemm_k_iterations_1 > 0; --gemm_k_iterations_1) {
        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations1; ++warp_mma_k) {
            if (warp_mma_k == Base::kWarpGemmIterations1 - 1) {
                 this->smem_iterator_B1_.store(tb_frag_B1);
                __syncthreads();
                 ++smem_iterator_B1_;
                if ( smem_write_stage_idx == 1 ) {
                    smem_iterator_B1_.add_tile_offset({-Base::Stage1, 0});
                }
                else {
                    this->warp_tile_iterator_B1_.add_tile_offset(
                    {-Base::Stage1 * Policy1::kPartitionsK *
                    Base::kWarpGemmIterations1,
                    0});
                }
                smem_write_stage_idx ^= 1;
            }
            this->warp_tile_iterator_B1_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations1);
            warp_tile_iterator_A1_.load(warp_frag_A1[(warp_mma_k + 1) % 2]);
            this->warp_tile_iterator_B1_.load(warp_frag_B1[(warp_mma_k + 1) % 2]);
            ++warp_tile_iterator_A1_;
            ++this->warp_tile_iterator_B1_;
             if (warp_mma_k == 0) {
                iterator_B1.load(tb_frag_B1);
                ++iterator_B1;
                if (gemm_k_iterations_1 <= 2) {
                    iterator_B1.clear_mask();
                }
            }
            warp_mma1(accum1, warp_frag_A1[warp_mma_k % 2], warp_frag_B1[warp_mma_k % 2], accum1);
        }
    }


// 3 Gemm    /// Iterator to load a warp-scoped tile of A1 operand from intermediate accumulator tile
    FragmentC1 after_epilogue_accu1;
    epilogue_1(output_op_1, accum1, after_epilogue_accu1, iterator_C1);
    FragmentIteratorA2 warp_tile_iterator_A2_(after_epilogue_accu1);
    FragmentB2 tb_frag_B2;
    tb_frag_B2.clear();
    iterator_B2.load(tb_frag_B2);
    ++iterator_B2;
    this->smem_iterator_B2_.store(tb_frag_B2);
    ++this->smem_iterator_B2_;
    __syncthreads();
    WarpFragmentA2 warp_frag_A2[2];
    WarpFragmentB2 warp_frag_B2[2];
    this->warp_tile_iterator_B2_.set_kgroup_index(0);
    warp_tile_iterator_A2_.load(warp_frag_A2[0]);
    this->warp_tile_iterator_B2_.load(warp_frag_B2[0]);
    ++warp_tile_iterator_A2_;
    ++this->warp_tile_iterator_B2_;
    Operator2 warp_mma2;
    smem_write_stage_idx = 1;
    int gemm_k_iterations_2 = FragmentIteratorA2::Policy::kIterations / Base::kWarpGemmIterations2;
    if (gemm_k_iterations_2 <= 1 ){
        iterator_B2.clear_mask();
    }
    CUTLASS_PRAGMA_UNROLL
    for (; gemm_k_iterations_2 > 0; --gemm_k_iterations_2) {
        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations2; ++warp_mma_k) {
            if (warp_mma_k == Base::kWarpGemmIterations2 - 1) {
                 this->smem_iterator_B2_.store(tb_frag_B2);
                __syncthreads();
                 ++smem_iterator_B2_;
                if ( smem_write_stage_idx == 1 ) {
                    smem_iterator_B2_.add_tile_offset({-Base::Stage2, 0});
                }
                else {
                    this->warp_tile_iterator_B2_.add_tile_offset(
                    {-Base::Stage2 * Policy2::kPartitionsK *
                    Base::kWarpGemmIterations2,
                    0});
                }
                smem_write_stage_idx ^= 1;
            }
            this->warp_tile_iterator_B2_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations2);
            warp_tile_iterator_A2_.load(warp_frag_A2[(warp_mma_k + 1) % 2]);
            this->warp_tile_iterator_B2_.load(warp_frag_B2[(warp_mma_k + 1) % 2]);
            ++warp_tile_iterator_A2_;
            ++this->warp_tile_iterator_B2_;
             if (warp_mma_k == 0) {
                iterator_B2.load(tb_frag_B2);
                ++iterator_B2;
                if (gemm_k_iterations_2 <= 2) {
                    iterator_B2.clear_mask();
                }
            }
            warp_mma2(accum2, warp_frag_A2[warp_mma_k % 2], warp_frag_B2[warp_mma_k % 2], accum2);
        }
    }


}
CUTLASS_DEVICE
B2bMmaPipelined(
    typename Base::B2bMmaSharedStorage &shared_storage,
    int thread_idx,
    int warp_idx,
    int lane_idx
):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A_(shared_storage.sharedStorage0.operand_A_ref(), thread_idx),
smem_iterator_B0_(shared_storage.sharedStorage0.operand_B_ref(), thread_idx),
smem_iterator_B1_(shared_storage.sharedStorage1.operand_B_ref(), thread_idx),
smem_iterator_B2_(shared_storage.sharedStorage2.operand_B_ref(), thread_idx) {
    int warp_idx_mn = warp_idx % (Base::WarpCount0::kM * Base::WarpCount0::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount0::kM * Base::WarpCount0::kN);
    int warp_idx_m = warp_idx_mn % Base::WarpCount0::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount0::kM;
    int tile_offset_k0 = Base::kWarpGemmIterations0 * warp_idx_k;
    int tile_offset_k1 = Base::kWarpGemmIterations1 * warp_idx_k;
    int tile_offset_k2 = Base::kWarpGemmIterations2 * warp_idx_k;
    this->warp_tile_iterator_A0_.add_tile_offset({warp_idx_m, tile_offset_k0});
    this->warp_tile_iterator_B0_.add_tile_offset({tile_offset_k0, warp_idx_n});
    this->warp_tile_iterator_B1_.add_tile_offset({tile_offset_k1, warp_idx_n});
    this->warp_tile_iterator_B2_.add_tile_offset({tile_offset_k2, warp_idx_n});
}
}; // class B2bMmaPipelined
} // namespace threadblock
} // namespace gemm
} // namespace cutlass
