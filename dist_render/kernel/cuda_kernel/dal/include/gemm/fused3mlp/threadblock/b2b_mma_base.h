
#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
namespace cutlass {
namespace gemm {
namespace threadblock {
template <
    typename Shape0_ ,
    typename Shape1_ ,
    typename Shape2_ ,
    typename Policy0_ ,
    typename Policy1_ ,
    typename Policy2_ ,
    int Stage0_ ,
    int Stage1_ ,
    int Stage2_ >
class B2bMmaBase{
public:
    using Shape0 = Shape0_ ;
    using Shape1 = Shape1_ ;
    using Shape2 = Shape2_ ;
    using Policy0 = Policy0_ ;
    using Policy1 = Policy1_ ;
    using Policy2 = Policy2_ ;
    static int const Stage0 = Stage0_ ;
    static int const Stage1 = Stage1_ ;
    static int const Stage2 = Stage2_ ;
using Operator0 = typename Policy0::Operator;
using Operator1 = typename Policy1::Operator;
using Operator2 = typename Policy2::Operator;
using WarpGemm0 = typename Policy0::Operator::Shape;
using WarpGemm1 = typename Policy1::Operator::Shape;
using WarpGemm2 = typename Policy2::Operator::Shape;
using WarpCount0 = GemmShape<Shape0::kM / WarpGemm0::kM, Shape0::kN / WarpGemm0::kN, Shape0::kK / WarpGemm0::kK>;
using WarpCount1 = GemmShape<Shape1::kM / WarpGemm1::kM, Shape1::kN / WarpGemm1::kN, Shape1::kK / WarpGemm1::kK>;
using WarpCount2 = GemmShape<Shape2::kM / WarpGemm2::kM, Shape2::kN / WarpGemm2::kN, Shape2::kK / WarpGemm2::kK>;
static int const kWarpGemmIterations0 = (WarpGemm0::kK / Operator0::Policy::MmaShape::kK);
static int const kWarpGemmIterations1 = (WarpGemm1::kK / Operator1::Policy::MmaShape::kK);
static int const kWarpGemmIterations2 = (WarpGemm2::kK / Operator2::Policy::MmaShape::kK);
 template<
    typename Shape_,
    typename Policy_,
    int ThisStage_
>
class SharedStorage {
public:
    using Shape = Shape_;
    using Policy = Policy_;
    static int const ThisStage = ThisStage_;
    using Operator = typename Policy::Operator;
        using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;
        /// Tensor reference to the B operand
    using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

    /// Shape of the A matrix operand in shared memory
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * ThisStage +
                                   Policy::SmemPaddingA::kColumn>;

    /// Shape of the B matrix operand in shared memory
    using ShapeB =
        MatrixShape<Shape::kK * ThisStage + Policy::SmemPaddingB::kRow,
                    Shape::kN + Policy::SmemPaddingB::kColumn>;

   public:

    /// Buffer for A operand
    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

    /// Buffer for B operand
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

   public:

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA()};
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    void * get_B_Shared_ptr() {
      return operand_B.data();
    }
  };
using SharedStorage0 = SharedStorage<Shape0, Policy0, Stage0>;
using SharedStorage1 = SharedStorage<Shape1, Policy1, Stage1>;
using SharedStorage2 = SharedStorage<Shape2, Policy2, Stage2>;
union B2bMmaSharedStorage {
    SharedStorage0 sharedStorage0;
    SharedStorage1 sharedStorage1;
    SharedStorage2 sharedStorage2;
};
void * C0_smm_ptr;
void * C1_smm_ptr;

protected:
typename Operator0::IteratorA warp_tile_iterator_A0_;
typename Operator0::IteratorB warp_tile_iterator_B0_;
typename Operator1::IteratorB warp_tile_iterator_B1_;
typename Operator2::IteratorB warp_tile_iterator_B2_;

public:
CUTLASS_DEVICE
B2bMmaBase(
    B2bMmaSharedStorage & shared_storage,
    int thread_idx,
    int warp_idx,
    int lane_idx
):
 warp_tile_iterator_A0_(shared_storage.sharedStorage0.operand_A_ref(), lane_idx),
 warp_tile_iterator_B0_(shared_storage.sharedStorage0.operand_B_ref(), lane_idx),
 warp_tile_iterator_B1_(shared_storage.sharedStorage1.operand_B_ref(), lane_idx),
 warp_tile_iterator_B2_(shared_storage.sharedStorage2.operand_B_ref(), lane_idx)
{
    C0_smm_ptr = shared_storage.sharedStorage0.get_B_Shared_ptr();
    C1_smm_ptr = shared_storage.sharedStorage1.get_B_Shared_ptr();
}
}; // class B2bMmaBase
} // namespace threadblock
} // namespace gemm
} // namespace cutlass
