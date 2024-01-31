
/* Auto Generated code - Do not edit.*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"

#include "gemm/fused3mlp/threadblock/b2b_mma_pipelined.h"
#include "gemm/fused3mlp/fixed_impl/epilogue/threadblock/fused_bias_act_epilogue.h"
#include "gemm/fused3mlp/fixed_impl/epilogue/threadblock/default_bias_act_epilogue_tensor_op.h"
#include "gemm/fused3mlp/fixed_impl/gemm/warp/mma_tensor_op_fragment_iterator_without_output_op.h"
namespace cutlass {
namespace gemm {
namespace threadblock {
template <
    typename ElementA_ ,
    typename LayoutA_ ,
    typename ElementB0_ ,
    typename LayoutB0_ ,
    typename ElementC0_ ,
    typename LayoutC0_ ,
    typename ElementAccumulator0_ ,
    typename EpilogueOutputOp0_ ,
    typename ThreadblockShape0_ ,
    typename WarpShape0_ ,
    typename ElementB1_ ,
    typename LayoutB1_ ,
    typename ElementC1_ ,
    typename LayoutC1_ ,
    typename ElementAccumulator1_ ,
    typename EpilogueOutputOp1_ ,
    typename ThreadblockShape1_ ,
    typename WarpShape1_ ,
    typename ElementB2_ ,
    typename LayoutB2_ ,
    typename ElementC2_ ,
    typename LayoutC2_ ,
    typename ElementAccumulator2_ ,
    typename EpilogueOutputOp2_ ,
    typename ThreadblockShape2_ ,
    typename WarpShape2_ ,
    typename ElementD_ ,
    typename LayoutD_ ,
    typename InstructionShape_ ,
    typename OperatorClass_ ,
    typename ArchTag_ ,
    typename ThreadblockSwizzle_ ,
    int Stages0_ ,
    int Stages1_ ,
    int Stages2_ ,
    int AlignmentA_ ,
    int AlignmentB_ ,
    bool SplitKSerial_ ,
    typename Operator_ ,
    bool IsBetaZero_ >
struct DefaultB2bMma{
public:
    using ElementA = ElementA_ ;
    using LayoutA = LayoutA_ ;
    using ElementB0 = ElementB0_ ;
    using LayoutB0 = LayoutB0_ ;
    using ElementC0 = ElementC0_ ;
    using LayoutC0 = LayoutC0_ ;
    using ElementAccumulator0 = ElementAccumulator0_ ;
    using EpilogueOutputOp0 = EpilogueOutputOp0_ ;
    using ThreadblockShape0 = ThreadblockShape0_ ;
    using WarpShape0 = WarpShape0_ ;
    using ElementB1 = ElementB1_ ;
    using LayoutB1 = LayoutB1_ ;
    using ElementC1 = ElementC1_ ;
    using LayoutC1 = LayoutC1_ ;
    using ElementAccumulator1 = ElementAccumulator1_ ;
    using EpilogueOutputOp1 = EpilogueOutputOp1_ ;
    using ThreadblockShape1 = ThreadblockShape1_ ;
    using WarpShape1 = WarpShape1_ ;
    using ElementB2 = ElementB2_ ;
    using LayoutB2 = LayoutB2_ ;
    using ElementC2 = ElementC2_ ;
    using LayoutC2 = LayoutC2_ ;
    using ElementAccumulator2 = ElementAccumulator2_ ;
    using EpilogueOutputOp2 = EpilogueOutputOp2_ ;
    using ThreadblockShape2 = ThreadblockShape2_ ;
    using WarpShape2 = WarpShape2_ ;
    using ElementD = ElementD_ ;
    using LayoutD = LayoutD_ ;
    using InstructionShape = InstructionShape_ ;
    using OperatorClass = OperatorClass_ ;
    using ArchTag = ArchTag_ ;
    using ThreadblockSwizzle = ThreadblockSwizzle_ ;
    static int const Stages0 = Stages0_ ;
    static int const Stages1 = Stages1_ ;
    static int const Stages2 = Stages2_ ;
    static int const AlignmentA = AlignmentA_ ;
    static int const AlignmentB = AlignmentB_ ;
    static bool const SplitKSerial = SplitKSerial_ ;
    using Operator = Operator_ ;
    static bool const IsBetaZero = IsBetaZero_ ;
}; // struct DefaultB2bMma
template <
    typename ElementA_ ,
    typename LayoutA_ ,
    typename ElementB0_ ,
    typename LayoutB0_ ,
    typename ElementC0_ ,
    typename LayoutC0_ ,
    typename ElementAccumulator0_ ,
    typename EpilogueOutputOp0_ ,
    typename ThreadblockShape0_ ,
    typename WarpShape0_ ,
    typename ElementB1_ ,
    typename LayoutB1_ ,
    typename ElementC1_ ,
    typename LayoutC1_ ,
    typename ElementAccumulator1_ ,
    typename EpilogueOutputOp1_ ,
    typename ThreadblockShape1_ ,
    typename WarpShape1_ ,
    typename ElementB2_ ,
    typename LayoutB2_ ,
    typename ElementC2_ ,
    typename LayoutC2_ ,
    typename ElementAccumulator2_ ,
    typename EpilogueOutputOp2_ ,
    typename ThreadblockShape2_ ,
    typename WarpShape2_ ,
    typename ElementD_ ,
    typename InstructionShape_ ,
    typename OperatorClass_ ,
    typename ArchTag_ ,
    typename ThreadblockSwizzle_ ,
    int Stages0_ ,
    int Stages1_ ,
    int Stages2_ ,
    int AlignmentA_ ,
    int AlignmentB_ ,
    bool SplitKSerial_ ,
    typename Operator_ ,
    bool IsBetaZero_ >
struct DefaultB2bMma<ElementA_,
LayoutA_,
ElementB0_,
LayoutB0_,
ElementC0_,
LayoutC0_,
ElementAccumulator0_,
EpilogueOutputOp0_,
ThreadblockShape0_,
WarpShape0_,
ElementB1_,
LayoutB1_,
ElementC1_,
LayoutC1_,
ElementAccumulator1_,
EpilogueOutputOp1_,
ThreadblockShape1_,
WarpShape1_,
ElementB2_,
LayoutB2_,
ElementC2_,
LayoutC2_,
ElementAccumulator2_,
EpilogueOutputOp2_,
ThreadblockShape2_,
WarpShape2_,
ElementD_,
cutlass::layout::RowMajor,
InstructionShape_,
OperatorClass_,
ArchTag_,
ThreadblockSwizzle_,
Stages0_,
Stages1_,
Stages2_,
AlignmentA_,
AlignmentB_,
SplitKSerial_,
Operator_,
IsBetaZero_
>{
public:
    using ElementA = ElementA_ ;
    using LayoutA = LayoutA_ ;
    using ElementB0 = ElementB0_ ;
    using LayoutB0 = LayoutB0_ ;
    using ElementC0 = ElementC0_ ;
    using LayoutC0 = LayoutC0_ ;
    using ElementAccumulator0 = ElementAccumulator0_ ;
    using EpilogueOutputOp0 = EpilogueOutputOp0_ ;
    using ThreadblockShape0 = ThreadblockShape0_ ;
    using WarpShape0 = WarpShape0_ ;
    using ElementB1 = ElementB1_ ;
    using LayoutB1 = LayoutB1_ ;
    using ElementC1 = ElementC1_ ;
    using LayoutC1 = LayoutC1_ ;
    using ElementAccumulator1 = ElementAccumulator1_ ;
    using EpilogueOutputOp1 = EpilogueOutputOp1_ ;
    using ThreadblockShape1 = ThreadblockShape1_ ;
    using WarpShape1 = WarpShape1_ ;
    using ElementB2 = ElementB2_ ;
    using LayoutB2 = LayoutB2_ ;
    using ElementC2 = ElementC2_ ;
    using LayoutC2 = LayoutC2_ ;
    using ElementAccumulator2 = ElementAccumulator2_ ;
    using EpilogueOutputOp2 = EpilogueOutputOp2_ ;
    using ThreadblockShape2 = ThreadblockShape2_ ;
    using WarpShape2 = WarpShape2_ ;
    using ElementD = ElementD_ ;
    using InstructionShape = InstructionShape_ ;
    using OperatorClass = OperatorClass_ ;
    using ArchTag = ArchTag_ ;
    using ThreadblockSwizzle = ThreadblockSwizzle_ ;
    static int const Stages0 = Stages0_ ;
    static int const Stages1 = Stages1_ ;
    static int const Stages2 = Stages2_ ;
    static int const AlignmentA = AlignmentA_ ;
    static int const AlignmentB = AlignmentB_ ;
    static bool const SplitKSerial = SplitKSerial_ ;
    using Operator = Operator_ ;
    static bool const IsBetaZero = IsBetaZero_ ;
using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape0, WarpShape0, InstructionShape, ElementA, LayoutA, ElementB0, LayoutB0, ElementAccumulator0, layout::RowMajor, OperatorClass, 2, Operator>;
using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape1, WarpShape1, InstructionShape, ElementA, LayoutA, ElementB1, LayoutB1, ElementAccumulator1, layout::RowMajor, OperatorClass, 2, Operator>;
using MmaCore2 = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape2, WarpShape2, InstructionShape, ElementA, LayoutA, ElementB2, LayoutB2, ElementAccumulator2, layout::RowMajor, OperatorClass, 2, Operator>;
using IteratorA0 = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore0::Shape::kM, MmaCore0::Shape::kK>, ElementA, LayoutA, 1, typename MmaCore0::IteratorThreadMapA, AlignmentA_>;
using IteratorB0 = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore0::Shape::kK, MmaCore0::Shape::kN>, ElementB0, LayoutB0, 0, typename MmaCore0::IteratorThreadMapB, AlignmentB_>;
using IteratorB1 = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore1::Shape::kK, MmaCore1::Shape::kN>, ElementB1, LayoutB1, 0, typename MmaCore1::IteratorThreadMapB, AlignmentB_>;
using IteratorB2 = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore2::Shape::kK, MmaCore2::Shape::kN>, ElementB2, LayoutB2, 0, typename MmaCore2::IteratorThreadMapB, AlignmentB_>;
using AccumulatorLayout = cutlass::layout::ColumnMajor;
using FragmentIteratorA1 = cutlass::gemm::warp::MmaTensorOpPureFragmentIterator<cutlass::MatrixShape<MmaCore1::WarpShape::kM, MmaCore1::InstructionShape::kK>, cutlass::MatrixShape<MmaCore0::WarpShape::kM, MmaCore0::WarpShape::kN>, MmaCore1::Shape::kK, ElementAccumulator0, ElementA, AccumulatorLayout, InstructionShape_, true>;
using FragmentIteratorA2 = cutlass::gemm::warp::MmaTensorOpPureFragmentIterator<cutlass::MatrixShape<MmaCore2::WarpShape::kM, MmaCore2::InstructionShape::kK>, cutlass::MatrixShape<MmaCore1::WarpShape::kM, MmaCore1::WarpShape::kN>, MmaCore2::Shape::kK, ElementAccumulator1, ElementA, AccumulatorLayout, InstructionShape_, true>;
using FusedAddBiasEpilogue0 = typename cutlass::epilogue::threadblock::DefaultFusedBiasActEpilogueTensorOp<ThreadblockShape0,typename MmaCore0::MmaPolicy::Operator, 1, EpilogueOutputOp0, 2>::Epilogue;
using FusedAddBiasEpilogue1 = typename cutlass::epilogue::threadblock::DefaultFusedBiasActEpilogueTensorOp<ThreadblockShape1,typename MmaCore1::MmaPolicy::Operator, 1, EpilogueOutputOp1, 2>::Epilogue;
using ThreadblockB2bMma = cutlass::gemm::threadblock::B2bMmaPipelined<typename MmaCore0::Shape, IteratorA0, typename MmaCore0::SmemIteratorA, IteratorB0, typename MmaCore0::SmemIteratorB, typename MmaCore1::Shape, FragmentIteratorA1, IteratorB1, typename MmaCore1::SmemIteratorB, typename MmaCore2::Shape, FragmentIteratorA2, IteratorB2, typename MmaCore2::SmemIteratorB, ElementAccumulator0, layout::RowMajor, EpilogueOutputOp0, EpilogueOutputOp1, FusedAddBiasEpilogue0, FusedAddBiasEpilogue1, typename MmaCore0::MmaPolicy, typename MmaCore1::MmaPolicy, typename MmaCore2::MmaPolicy, Stages0, Stages1, Stages2>;
}; // struct DefaultB2bMma
} // namespace threadblock
} // namespace gemm
} // namespace cutlass
