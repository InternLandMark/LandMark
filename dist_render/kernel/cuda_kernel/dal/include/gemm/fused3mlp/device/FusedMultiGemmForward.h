/* Auto Generated code - Do not edit.*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "gemm/fused3mlp/kernel/b2b_gemm.h"
#include "gemm/fused3mlp/kernel/default_b2b_gemm.h"

namespace cutlass {
namespace gemm {
namespace device {
template <
    typename ElementA_ = cutlass::half_t ,
    typename LayoutA_ = cutlass::layout::RowMajor ,
    typename ElementB0_ = cutlass::half_t ,
    typename LayoutB0_ = cutlass::layout::ColumnMajor ,
    typename ElementC0_ = cutlass::half_t ,
    typename LayoutC0_ = cutlass::layout::RowMajor ,
    typename ElementAccumulator0_ = cutlass::half_t ,
    typename EpilogueOutputOp0_ = cutlass::epilogue::thread::LinearCombinationRelu<ElementC0_, 32, ElementAccumulator0_, ElementAccumulator0_, cutlass::epilogue::thread::ScaleType::NoBetaScaling> ,
    typename ThreadblockShape0_ = cutlass::gemm::GemmShape<128, 128, 32> ,
    typename WarpShape0_ = cutlass::gemm::GemmShape<32, 128, 32> ,
    typename ElementB1_ = cutlass::half_t ,
    typename LayoutB1_ = cutlass::layout::ColumnMajor ,
    typename ElementC1_ = cutlass::half_t ,
    typename LayoutC1_ = cutlass::layout::RowMajor ,
    typename ElementAccumulator1_ = cutlass::half_t ,
    typename EpilogueOutputOp1_ = cutlass::epilogue::thread::LinearCombinationRelu<ElementC0_, 32, ElementAccumulator0_, ElementAccumulator0_, cutlass::epilogue::thread::ScaleType::NoBetaScaling> ,
    typename ThreadblockShape1_ = cutlass::gemm::GemmShape<128, 128, 32> ,
    typename WarpShape1_ = cutlass::gemm::GemmShape<32, 128, 32> ,
    typename ElementB2_ = cutlass::half_t ,
    typename LayoutB2_ = cutlass::layout::ColumnMajor ,
    typename ElementC2_ = cutlass::half_t ,
    typename LayoutC2_ = cutlass::layout::RowMajor ,
    typename ElementAccumulator2_ = cutlass::half_t ,
    typename EpilogueOutputOp2_ = cutlass::epilogue::thread::LinearCombinationSigmoid<ElementC0_, 8, ElementAccumulator0_, ElementAccumulator0_, cutlass::epilogue::thread::ScaleType::NoBetaScaling> ,
    typename ThreadblockShape2_ = cutlass::gemm::GemmShape<128, 32, 32> ,
    typename WarpShape2_ = cutlass::gemm::GemmShape<32, 32, 32> ,
    typename ElementD_ = cutlass::half_t ,
    typename LayoutD_ = cutlass::layout::RowMajor ,
    typename InstructionShape_ = cutlass::gemm::GemmShape<16, 8, 8> ,
    typename OperatorClass_ = arch::OpClassTensorOp ,
    typename ArchTag_ = cutlass::arch::Sm75 ,
    typename ThreadblockSwizzle_ = threadblock::GemmBatchedIdentityThreadblockSwizzle ,
    int Stages0_ = 2 ,
    int Stages1_ = 2 ,
    int Stages2_ = 2 ,
    int AlignmentA_ = 8 ,
    int AlignmentB_ = 8 ,
    bool SplitKSerial_ = false ,
    typename Operator_ = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB0_, ElementC0_, ElementAccumulator0_>::Operator ,
    bool IsBetaZero_ = false >
class FusedMultiGemmForward{
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
using B2bGemmKernel = typename kernel::DefaultB2bGemm<
    ElementA,
    LayoutA,
    ElementB0,
    LayoutB0,
    ElementC0,
    LayoutC0,
    ElementAccumulator0,
    EpilogueOutputOp0,
    ThreadblockShape0,
    WarpShape0,
    ElementB1,
    LayoutB1,
    ElementC1,
    LayoutC1,
    ElementAccumulator1,
    EpilogueOutputOp1,
    ThreadblockShape1,
    WarpShape1,
    ElementB2,
    LayoutB2,
    ElementC2,
    LayoutC2,
    ElementAccumulator2,
    EpilogueOutputOp2,
    ThreadblockShape2,
    WarpShape2,
    ElementD,
    LayoutD,
    InstructionShape,
    OperatorClass,
    ArchTag,
    ThreadblockSwizzle,
    Stages0,
    Stages1,
    Stages2,
    AlignmentA,
    AlignmentB,
    SplitKSerial,
    Operator,
    IsBetaZero_
>::B2bGemmKernel;


struct Arguments{
    GemmCoord problem_size_0;
    GemmCoord problem_size_1;
    GemmCoord problem_size_2;
    TensorRef<ElementA const, LayoutA> ref_A0;
    TensorRef<ElementB0 const, LayoutB0> ref_B0;
    TensorRef<ElementC0 const, LayoutC0> ref_C0;
    TensorRef<ElementB1 const, LayoutB1> ref_B1;
    TensorRef<ElementC1 const, LayoutC1> ref_C1;
    TensorRef<ElementB2 const, LayoutB2> ref_B2;
    TensorRef<ElementC2 const, LayoutC2> ref_C2;
    TensorRef<ElementD, LayoutD> ref_D2;
    typename EpilogueOutputOp0::Params epilogue0;
    typename EpilogueOutputOp1::Params epilogue1;
    typename EpilogueOutputOp2::Params epilogue2;
    int batch_count;
    CUTLASS_HOST_DEVICE
    Arguments (): problem_size_0(0,0,0),problem_size_1(0,0,0),problem_size_2(0,0,0){ }

    CUTLASS_HOST_DEVICE
    Arguments (
    GemmCoord problem_size_0_,
    GemmCoord problem_size_1_,
    GemmCoord problem_size_2_,
    TensorRef<ElementA const, LayoutA> ref_A0_,
    TensorRef<ElementB0 const, LayoutB0> ref_B0_,
    TensorRef<ElementC0 const, LayoutC0> ref_C0_,
    TensorRef<ElementB1 const, LayoutB1> ref_B1_,
    TensorRef<ElementC1 const, LayoutC1> ref_C1_,
    TensorRef<ElementB2 const, LayoutB2> ref_B2_,
    TensorRef<ElementC2 const, LayoutC2> ref_C2_,
    TensorRef<ElementD, LayoutD> ref_D2_,
    typename EpilogueOutputOp0::Params epilogue0_,
    typename EpilogueOutputOp1::Params epilogue1_,
    typename EpilogueOutputOp2::Params epilogue2_,
    int batch_count_
):
    problem_size_0(problem_size_0_),
    problem_size_1(problem_size_1_),
    problem_size_2(problem_size_2_),
    ref_A0(ref_A0_),
    ref_B0(ref_B0_),
    ref_C0(ref_C0_),
    ref_B1(ref_B1_),
    ref_C1(ref_C1_),
    ref_B2(ref_B2_),
    ref_C2(ref_C2_),
    ref_D2(ref_D2_),
    epilogue0(epilogue0_),
    epilogue1(epilogue1_),
    epilogue2(epilogue2_),
    batch_count(batch_count_) { }


}; // struct Arguments

FusedMultiGemmForward() {}
Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
// Determine grid shape
ThreadblockSwizzle threadblock_swizzle;
cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
  args.problem_size_0,
  { ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK },
  args.batch_count);
// Initialize the Params structure
params_ = typename B2bGemmKernel::Params{
  args.problem_size_0,
  args.problem_size_1,
  args.problem_size_2,
  grid_shape,
  args.ref_A0.non_const_ref(),
  args.ref_B0.non_const_ref(),
  args.ref_C0.non_const_ref(),
  args.ref_B1.non_const_ref(),
  args.ref_C1.non_const_ref(),
  args.ref_B2.non_const_ref(),
  args.ref_C2.non_const_ref(),
  args.ref_D2,
  args.epilogue0,
  args.epilogue1,
  args.epilogue2,
  args.batch_count
};
return Status::kSuccess;
}

Status run(cudaStream_t stream = nullptr) {

  ThreadblockSwizzle threadblock_swizzle;

  dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
  dim3 block(B2bGemmKernel::kThreadCount, 1, 1);

  cudaError_t result;

  int smem_size = int(sizeof(typename B2bGemmKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    result = cudaFuncSetAttribute(Kernel<B2bGemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    if (result != cudaSuccess) {
      return Status::kErrorInternal;
    }

    result = cudaFuncSetAttribute(
        Kernel<B2bGemmKernel>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    if (result != cudaSuccess) {
      return Status::kErrorInternal;
    }
  }
  cutlass::Kernel<B2bGemmKernel><<<grid, block, smem_size, stream>>>(params_);
  result = cudaGetLastError();
  return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

Status operator()(
  Arguments const &args,
  void *workspace = nullptr,
  cudaStream_t stream = nullptr) {
  Status status = initialize(args, workspace);

  if (status == Status::kSuccess) {
    status = run(stream);
  }
  return status;
}

Status operator()(
  cudaStream_t stream = nullptr) {
   Status status = run(stream);
   return status;
}
private:
 typename B2bGemmKernel::Params params_;
}; // class FusedMultiGemmForward
} // namespace device
} // namespace gemm
} // namespace cutlass
