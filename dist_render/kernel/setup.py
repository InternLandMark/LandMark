import os
from distutils.sysconfig import get_python_inc

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

kernel_dir = "./cuda_kernel/"
dal_kernel_dir = "./cuda_kernel/dal/"
python_env_include_path = get_python_inc()
cutlass_dir = os.environ.get(
    "CUTLASS_DIR",
    "/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/"
    "checkpoint/V2.0.0/kernel/leftsmall/dependecies/cutlass/",
)
print("python_env_include_path:", python_env_include_path)
print("cutlass_dir:", cutlass_dir)

cutlass_include = os.path.join(cutlass_dir, "include")
cutlass_tools_include = os.path.join(cutlass_dir, "tools/util/include")
leftsmall_include = os.path.join(cutlass_dir, "../../include")
cuda_kernel_include = os.path.join(os.getcwd(), "cuda_kernel/dal/include")


setup(
    name="SamplerayGridsample_dal",
    ext_modules=[
        CUDAExtension(
            "SamplerayGridsample_dal",
            sources=[
                dal_kernel_dir + "sampleray_gridsample/compute_sampleray_gridsample_cuda.cpp",
                dal_kernel_dir + "sampleray_gridsample/compute_caltvalmin_cuda_kernel.cu",
                dal_kernel_dir + "sampleray_gridsample/compute_caltvalmin_sharedmem_cuda_kernel.cu",
                dal_kernel_dir
                + "sampleray_gridsample/"
                "compute_sampleray_withinhull_coord_trunc_precaltmin_validsamples_cuda_kernel_opt.cu",
                dal_kernel_dir + "sampleray_gridsample/compute_gridsample3D_2d_bool_dev_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="assign_blocks_to_samples_dal",
    ext_modules=[
        CUDAExtension(
            "AssignBlocksToSamples_dal",
            [
                dal_kernel_dir + "sampleray_gridsample/compute_assign_blocks_to_samples_cuda.cpp",
                dal_kernel_dir + "sampleray_gridsample/compute_assign_blocks_to_samples_cuda_kernel.cu",
                dal_kernel_dir + "sampleray_gridsample/compute_assign_blocks_to_samples_norm_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="compute_beta_dal",
    ext_modules=[
        CUDAExtension(
            "compute_beta_dal",
            [
                dal_kernel_dir + "volrend/compute_beta_cuda.cpp",
                dal_kernel_dir + "volrend/compute_beta_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

setup(
    name="compute_weight_dal",
    ext_modules=[
        CUDAExtension(
            "compute_weight_dal",
            [
                dal_kernel_dir + "volrend/compute_weight_cuda.cpp",
                dal_kernel_dir + "volrend/compute_weight_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

setup(
    name="grid_sampler_ndhwc_dal",
    ext_modules=[
        CUDAExtension(
            "grid_sampler_ndhwc_dal",
            sources=[
                dal_kernel_dir + "densityappfeature/grid_sampler_ndhwc.cpp",
                dal_kernel_dir + "densityappfeature/grid_sampler_ndhwc_cuda.cu",
            ],
            include_dirs=[
                python_env_include_path,
                cutlass_include,
                cutlass_tools_include,
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="pe_concate_dal",
    ext_modules=[
        CUDAExtension(
            "pe_concate_dal",
            sources=[
                dal_kernel_dir + "pe_concate_mlp/pe_concate_half.cu",
            ],
            extra_compile_args={
                "cxx": ["-std=c++14 -g -O3"],
                "nvcc": ["-maxrregcount=16", "-use_fast_math", "-Xptxas", "-dlcm=ca"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="gemm_fp16_dal",
    ext_modules=[
        CUDAExtension(
            "gemm_fp16_dal",
            sources=[
                dal_kernel_dir + "pe_concate_mlp/cutlass/gemm_fp16.cu",
            ],
            include_dirs=[
                python_env_include_path,
                cutlass_include,
                cutlass_tools_include,
                cuda_kernel_include,
                leftsmall_include,
            ],
            extra_compile_args={
                "cxx": ["-std=c++17  -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="pe_dal",
    ext_modules=[
        CUDAExtension(
            "pe_column_major_half2half_dal",
            sources=[
                dal_kernel_dir + "pe_concate_mlp/column_major/pe_column_major_half2half.cu",
            ],
            include_dirs=[python_env_include_path, cuda_kernel_include, leftsmall_include],
            extra_compile_args={
                "cxx": ["-std=c++17 -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="expand_index_encoding_dal",
    ext_modules=[
        CUDAExtension(
            "expand_encoding_2half_dal",
            sources=[
                dal_kernel_dir + "pe_concate_mlp/column_major/expand_encoding_2half.cu",
            ],
            include_dirs=[python_env_include_path, cuda_kernel_include, leftsmall_include],
            extra_compile_args={
                "cxx": ["-std=c++17 -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="expand_index_encoding_mlp_dal",
    ext_modules=[
        CUDAExtension(
            "pipeline_expand_index_encoding_mlp_dal",
            sources=[
                dal_kernel_dir + "pe_concate_mlp/column_major/pipeline_expand_encoding_mlp.cu",
            ],
            include_dirs=[
                python_env_include_path,
                cuda_kernel_include,
                cutlass_include,
                cutlass_tools_include,
                leftsmall_include,
            ],
            extra_compile_args={
                "cxx": ["-std=c++17 -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

# v0606
setup(
    name="frequency_encoding_dal",
    ext_modules=[
        CUDAExtension(
            "frequency_encoding_dal",
            sources=[
                dal_kernel_dir + "pe_concate_mlp/frequency_encoding.cu",
            ],
            include_dirs=[cuda_kernel_include],
            extra_compile_args={
                "cxx": ["-std=c++17 -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

cuda_kernel_include = os.path.join(os.getcwd(), "cuda_kernel/include")


setup(
    name="SamplerayGridsample",
    ext_modules=[
        CUDAExtension(
            "SamplerayGridsample",
            sources=[
                kernel_dir + "sampleray_gridsample/compute_sampleray_gridsample_cuda.cpp",
                kernel_dir + "sampleray_gridsample/compute_caltvalmin_cuda_kernel.cu",
                kernel_dir + "sampleray_gridsample/compute_caltvalmin_sharedmem_cuda_kernel.cu",
                kernel_dir
                + "sampleray_gridsample/"
                "compute_sampleray_withinhull_coord_trunc_precaltmin_validsamples_cuda_kernel_opt.cu",
                kernel_dir + "sampleray_gridsample/compute_gridsample3D_2d_cuda_kernel.cu",
                kernel_dir + "sampleray_gridsample/compute_gridsample3D_2d_bounderror_cuda_kernel.cu",
                kernel_dir + "sampleray_gridsample/compute_gridsample3D_2d_dev_cuda_kernel.cu",
                kernel_dir + "sampleray_gridsample/compute_gridsample3D_2d_bool_dev_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="assign_blocks_to_samples",
    ext_modules=[
        CUDAExtension(
            "AssignBlocksToSamples",
            [
                kernel_dir + "sampleray_gridsample/compute_assign_blocks_to_samples_cuda.cpp",
                kernel_dir + "sampleray_gridsample/compute_assign_blocks_to_samples_cuda_kernel.cu",
                kernel_dir + "sampleray_gridsample/compute_assign_blocks_to_samples_norm_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="compute_beta",
    ext_modules=[
        CUDAExtension(
            "compute_beta",
            [
                kernel_dir + "volrend/compute_beta_cuda.cpp",
                kernel_dir + "volrend/compute_beta_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="compute_weight",
    ext_modules=[
        CUDAExtension(
            "compute_weight",
            [
                kernel_dir + "volrend/compute_weight_cuda.cpp",
                kernel_dir + "volrend/compute_weight_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="compute_grid_sample_and_ewproduct",
    ext_modules=[
        CUDAExtension(
            "compute_grid_sample_and_ewproduct",
            sources=[
                kernel_dir + "densityappfeature/compute_grid_sample_and_ewproduct_cuda.cpp",
                kernel_dir + "densityappfeature/compute_grid_sample_and_ewproduct_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="compute_appfeature",
    ext_modules=[
        CUDAExtension(
            "compute_appfeature",
            sources=[
                kernel_dir + "densityappfeature/compute_appfeature_cuda.cpp",
                kernel_dir + "densityappfeature/compute_appfeature_cuda_kernel.cu",
            ],
            include_dirs=[
                python_env_include_path,
                cutlass_include,
                cutlass_tools_include,
            ],
        ),
        CUDAExtension(
            "gemm_3xtf32_fast_accurate_GaColumnMajor",
            sources=[kernel_dir + "densityappfeature/ampere_3xtf32_fast_accurate_tensorop_gemm_GaColumnMajor.cu"],
            include_dirs=[
                python_env_include_path,
                cutlass_include,
                cutlass_tools_include,
                cuda_kernel_include,
            ],
            extra_compile_args={
                "cxx": ["-std=c++17 -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
        CUDAExtension(
            "cutlass_gemm_fp16_for_appfeature",
            sources=[kernel_dir + "densityappfeature/cutlass_gemm_fp16_for_appfeature.cu"],
            include_dirs=[
                python_env_include_path,
                cutlass_include,
                cutlass_tools_include,
                cuda_kernel_include,
            ],
            extra_compile_args={
                "cxx": ["-std=c++17  -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="pe_concate",
    ext_modules=[
        CUDAExtension(
            "pe_concate",
            sources=[
                kernel_dir + "pe_concate_mlp/pe_concate_half.cu",
            ],
            extra_compile_args={
                "cxx": ["-std=c++14 -g -O3"],
                "nvcc": ["-maxrregcount=16", "-use_fast_math", "-Xptxas", "-dlcm=ca"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="gemm_fp16",
    ext_modules=[
        CUDAExtension(
            "gemm_fp16",
            sources=[
                kernel_dir + "pe_concate_mlp/cutlass/gemm_fp16.cu",
            ],
            include_dirs=[
                python_env_include_path,
                cutlass_include,
                cutlass_tools_include,
                cuda_kernel_include,
            ],
            extra_compile_args={
                "cxx": ["-std=c++17  -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="pe",
    ext_modules=[
        CUDAExtension(
            "pe_column_major_half2half",
            sources=[
                kernel_dir + "pe_concate_mlp/column_major/pe_column_major_half2half.cu",
            ],
            include_dirs=[
                python_env_include_path,
                cuda_kernel_include,
            ],
            extra_compile_args={
                "cxx": ["-std=c++17 -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="expand_index_encoding",
    ext_modules=[
        CUDAExtension(
            "expand_encoding_2half",
            sources=[
                kernel_dir + "pe_concate_mlp/column_major/expand_encoding_2half.cu",
            ],
            include_dirs=[
                python_env_include_path,
                cuda_kernel_include,
            ],
            extra_compile_args={
                "cxx": ["-std=c++17 -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="expand_index_encoding_mlp",
    ext_modules=[
        CUDAExtension(
            "pipeline_expand_index_encoding_mlp",
            sources=[
                kernel_dir + "pe_concate_mlp/column_major/pipeline_expand_encoding_mlp.cu",
            ],
            include_dirs=[
                python_env_include_path,
                cuda_kernel_include,
                cutlass_include,
                cutlass_tools_include,
                cuda_kernel_include,
            ],
            extra_compile_args={
                "cxx": ["-std=c++17 -O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


setup(
    name="grid_sample_3d_ndhwc2ncdhw",
    ext_modules=[
        CUDAExtension(
            "grid_sample_3d_ndhwc2ncdhw",
            sources=[
                kernel_dir + "densityappfeature/grid_sampler_3d_ndhwc.cpp",
                kernel_dir + "densityappfeature/grid_sampler_3d_ndhwc_kernel.cu",
            ],
            include_dirs=[
                python_env_include_path,
                os.path.join(cutlass_dir + "dependecies/cutlass/include"),
                os.path.join(cutlass_dir + "dependecies/cutlass/tools/util/include"),
                os.path.join(cutlass_dir + "include"),
            ],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
