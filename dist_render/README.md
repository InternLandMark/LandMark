<p align="center">
<video width="400" height="300" src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/videos/1k.mp4"></video>
</p>

# 💻 About
The codes in this directory implement a distributed rendering system,
to support real-time(30~ms Latency, 30+FPS) rendering of large scale GridNeRF model(100km2 or bigger) with custom kernels and multiple parallel modes and other optimizations. Such as:
- The mixed-mode parallel computation of data parallel and tensor parallel.
- Pipeline optimization, to overlap preprocess and postprocess.
- Kernel fusion specified for GridNeRF model and mixed-mode parallel computation.
- Unique offloading to support infinite rendering area.(developing)

# 🎨 Features
This system supports distributed real-time rendering of the following models:
- GridNeRF model trained with sequential ways. We define the model type as `Torch`.
- GridNeRF model trained with sequential ways and rendering with kernel fusion optimization. We define the model type as `TorchKernelFusion`.(developing)
- GridNeRF model trained with branch parallel. We define the model type as `MultiBlockTorch`.
- GridNeRF model trained with branch parallel and rendering with kernel fusion optimization. We define the model type as `MultiBlockKernelFusion`.
- GridNeRF model trained with branch parallel and rendering with tensor parallel optimization of computer memory. We define the model type as `MultiBlockTensorParallelTorch`.
- GridNeRF model trained with branch parallel and rendering with tensor parallel optimization of computer memory and rendering with kernel fusion optimization. We define the model type as `MultiBlockTensorParallelKernelFusion`.

# 🚀 Quickstart
## Start Environment
You can refer to readme in root directory on how to create landmark environment.
```
source /your_landmark_env_path/env/landmark
```

## Prepare Dataset
Refer to readme in root directory

## Set Config Arguments
Most of the configuration arguments in config file are the same as the corresponding model configs described in root directory readme, but there are several specifial arguments for distributed rendering.
- branch_parallel: Whether the model is trained by branch parallel.
- plane_division: The plane division if the model is trained by branch parallel.
- render_batch_size: Chunk size of nerf rays, you can config this arg by `renderer.py --render_batch_size` also.
- alpha_mask_filter_thre: Filter samples with alpha that under the threshold, It influences rendering latency. You can config this arg by `renderer.py --alpha_mask_filter_thre` also.
- sampling_opt: Whether to use sampling optimization when rendering, it's 1 usually.
- ckpt: Pytorch ckpt model trained by trainer, you can config this arg by `renderer.py --ckpt` also.

## Set Environment Variables
- export RENDER_1080P=False: Set `True` if you want to render 1080p image.
- export PROFILE_TIMELINE=False: Set `True` if you want to save pytorch timeline file in main rank process.
- export TENSOR_PARALLEL_SIZE=4: Set tensor parallel group world size if you want to render model with tensor parallel optimization. Data parallel optimization is used by default. Model type configed by `renderer.py --model_type` should be `MultiBlockTensorParallelTorch` or `MultiBlockTensorParallelKernelFusion`.
- export HOME=/your_home
- export PYTHONPATH=/your_landmark_path/:$PYTHONPATH


## Compile Custom Kernels
If you want to use GridNeRF kernel fusion optimization, compile and install custom kernels firstly.
Our kernels rely on the third-party project `cutlass`, so you should clone and modify cutlass codes(it has error) in advance. See [here](https://github.com/InternLandMark/LandMark_Documentation/blob/main/kernel_docs/README.md) for help.
```
pip install ninja --user
pip uninstall pe pe-concate assign-blocks-to-samples compute-appfeature compute-beta compute-grid-sample-and-ewproduct compute-weight gemm-fp16 SamplerayGridsample expand-index-encoding expand-index-encoding-mlp
export PATH=/home/$YOUR_DIR/.local/bin:$PATH
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
export CUTLASS_DIR=/your_cutlass_dir/
cd ~/landmark/dist_render/kernel/
python setup.py install --user
```

## Render poses in dataset
Start running command after preparing distributed config file and ckpt model and appoint the corresponding model type for distributed rendering.if you want to save picture results in `dist_render/picture/` directory, config `--save_png` for `renderer.py`. See `scripts/` for detail.
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs --nnodes=number_of_nodes --node_rank=rank_of_node ./renderer.py --model_type=MultiBlockTorch --config=./confs/dist_render_conf/dist_render_city_multi_branch_encodeapp.txt --ckpt=/your_ckpt_model_path
```
