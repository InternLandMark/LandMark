
/* Auto Generated code - Do not edit.*/

#pragma once
#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "gemm/fused3mlp/kernel/b2b_gemm.h"
#include "gemm/fused3mlp/threadblock/default_b2b_mma.h"
namespace cutlass {
namespace gemm {
namespace kernel {
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
struct DefaultB2bGemm{
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
}; // struct DefaultB2bGemm
template <
    typename ElementA_ ,
    typename LayoutA_ ,
    typename ElementB0_ ,
    typename LayoutB0_ ,
    typename ElementC0_ ,
    typename ElementAccumulator0_ ,
    typename EpilogueOutputOp0_ ,
    typename ThreadblockShape0_ ,
    typename WarpShape0_ ,
    typename ElementB1_ ,
    typename LayoutB1_ ,
    typename ElementC1_ ,
    typename ElementAccumulator1_ ,
    typename EpilogueOutputOp1_ ,
    typename ThreadblockShape1_ ,
    typename WarpShape1_ ,
    typename ElementB2_ ,
    typename LayoutB2_ ,
    typename ElementC2_ ,
    typename ElementAccumulator2_ ,
    typename EpilogueOutputOp2_ ,
    typename ThreadblockShape2_ ,
    typename WarpShape2_ ,
    typename ElementD_ ,
    typename LayoutD_ ,
    typename InstructionShape_ ,
    typename ThreadblockSwizzle_ ,
    int Stages0_ ,
    int Stages1_ ,
    int Stages2_ ,
    int AlignmentA_ ,
    int AlignmentB_ ,
    bool SplitKSerial_ ,
    typename Operator_ ,
    bool IsBetaZero_ >
struct DefaultB2bGemm<ElementA_,
LayoutA_,
ElementB0_,
LayoutB0_,
ElementC0_,
layout::RowMajor,
ElementAccumulator0_,
EpilogueOutputOp0_,
ThreadblockShape0_,
WarpShape0_,
ElementB1_,
LayoutB1_,
ElementC1_,
layout::RowMajor,
ElementAccumulator1_,
EpilogueOutputOp1_,
ThreadblockShape1_,
WarpShape1_,
ElementB2_,
LayoutB2_,
ElementC2_,
layout::RowMajor,
ElementAccumulator2_,
EpilogueOutputOp2_,
ThreadblockShape2_,
WarpShape2_,
ElementD_,
LayoutD_,
InstructionShape_,
arch::OpClassTensorOp,
arch::Sm75,
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
    using ElementAccumulator0 = ElementAccumulator0_ ;
    using EpilogueOutputOp0 = EpilogueOutputOp0_ ;
    using ThreadblockShape0 = ThreadblockShape0_ ;
    using WarpShape0 = WarpShape0_ ;
    using ElementB1 = ElementB1_ ;
    using LayoutB1 = LayoutB1_ ;
    using ElementC1 = ElementC1_ ;
    using ElementAccumulator1 = ElementAccumulator1_ ;
    using EpilogueOutputOp1 = EpilogueOutputOp1_ ;
    using ThreadblockShape1 = ThreadblockShape1_ ;
    using WarpShape1 = WarpShape1_ ;
    using ElementB2 = ElementB2_ ;
    using LayoutB2 = LayoutB2_ ;
    using ElementC2 = ElementC2_ ;
    using ElementAccumulator2 = ElementAccumulator2_ ;
    using EpilogueOutputOp2 = EpilogueOutputOp2_ ;
    using ThreadblockShape2 = ThreadblockShape2_ ;
    using WarpShape2 = WarpShape2_ ;
    using ElementD = ElementD_ ;
    using LayoutD = LayoutD_ ;
    using InstructionShape = InstructionShape_ ;
    using ThreadblockSwizzle = ThreadblockSwizzle_ ;
    static int const Stages0 = Stages0_ ;
    static int const Stages1 = Stages1_ ;
    static int const Stages2 = Stages2_ ;
    static int const AlignmentA = AlignmentA_ ;
    static int const AlignmentB = AlignmentB_ ;
    static bool const SplitKSerial = SplitKSerial_ ;
    using Operator = Operator_ ;
    static bool const IsBetaZero = IsBetaZero_ ;
using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<
ElementA_,
LayoutA_,
ElementB0_,
LayoutB0_,
ElementC0_,
layout::RowMajor,
ElementAccumulator0_,
EpilogueOutputOp0_,
ThreadblockShape0_,
WarpShape0_,
ElementB1_,
LayoutB1_,
ElementC1_,
layout::RowMajor,
ElementAccumulator1_,
EpilogueOutputOp1_,
ThreadblockShape1_,
WarpShape1_,
ElementB2_,
LayoutB2_,
ElementC2_,
layout::RowMajor,
ElementAccumulator2_,
EpilogueOutputOp2_,
ThreadblockShape2_,
WarpShape2_,
ElementD_,
LayoutD_,
InstructionShape_,
arch::OpClassTensorOp,
arch::Sm75,
ThreadblockSwizzle_,
Stages0_,
Stages1_,
Stages2_,
AlignmentA_,
AlignmentB_,
SplitKSerial_,
Operator_,
IsBetaZero_
>::ThreadblockB2bMma;
static const int kPartitionsK2 = ThreadblockShape2::kK / WarpShape2::kK;
using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape2,
    typename B2bMma::Operator2,
    kPartitionsK2,
    EpilogueOutputOp2,
    EpilogueOutputOp2::kCount
>::Epilogue;
using B2bGemmKernel = kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle, SplitKSerial>;

}; // struct DefaultB2bGemm
} // namespace kernel
} // namespace gemm
} // namespace cutlass
