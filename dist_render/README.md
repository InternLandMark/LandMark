<p align="center">
<video width="400" height="300" src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/videos/1k.mp4"></video>
</p>

# 💻 About
The codes in this directory implement a distributed rendering system,
to support real-time(30~ms Latency, 30+FPS) rendering of large scale GridNeRF model(100km2 or bigger) with custom kernels and multiple parallel modes and other optimizations. 
# 🎨 Features
- The mixed-mode parallel computation of data parallel and tensor parallel.
- Pipeline optimization, to overlap preprocess and postprocess.
- Kernel fusion specified for GridNeRF model and mixed-mode parallel computation.
- [Dynamic fetching](https://github.com/InternLandMark/LandMark_Documentation#Dynamic-Fetching), a strategy to prefetch and offload parameters for limited GPU memory, to support infinite rendering area.



# 🚀 Quickstart
## Start Environment
You can refer to readme in root directory on how to create landmark environment.
```
source /your_landmark_env_path/env/landmark
```

## Prepare Dataset
Refer to readme in root directory
## Model Support
This system supports distributed real-time rendering of the following models:
- GridNeRF model trained with sequential ways. We define the model type as `Torch`.
- GridNeRF model trained with sequential ways and rendering with kernel fusion optimization. We define the model type as `TorchKernelFusion`.(developing)
- GridNeRF model trained with branch parallel. We define the model type as `MultiBlockTorch`.
- GridNeRF model trained with branch parallel and rendering with kernel fusion optimization. We define the model type as `MultiBlockKernelFusion`.
- GridNeRF model trained with branch parallel and rendering with tensor parallel optimization of computer memory. We define the model type as `MultiBlockTensorParallelTorch`.
- GridNeRF model trained with branch parallel and rendering with tensor parallel optimization of computer memory and rendering with kernel fusion optimization. We define the model type as `MultiBlockTensorParallelKernelFusion`.
- GridNeRF model with dynamic fetching.We define the model type as `MovingAreaTorch`.
## Set Arguments
### Set Startup Parameter
Startup Parameter is set after `renderer.py`,like `renderer.py --model_type YOUR_MODEL_TYPE`.
- model_type: Which model is selected for rendering.Refer to model types listed above.
- use_multistream: Whether use multiple streams with preprocess and postprocess for pipeline optimization.
- alpha_mask_filter_thre: Filter samples with alpha that under the threshold, It influences rendering latency. 
- render_batch_size: Chunk size of nerf rays.
- ckpt: Pytorch ckpt model trained by trainer.
- save_png: Whether to save picture results in `dist_render/picture/` directory.

See `renderer.py` for all startup parameters.
### Set Config Files
The following parameters could be set in the config file. Some of the startup parameters could also be set in the config file if they are less likely to change each time for you.
- branch_parallel: Whether the model is trained by branch parallel.
- plane_division: The plane division if the model is trained by branch parallel.
- sampling_opt: Whether to use sampling optimization when rendering, it's 1 usually.
- dynamic_fetching: Whether to use dynamic fetching when rendering.Make sure compile kernels before dynamic fetching. Corresponding `model_type`  should be `MovingAreaTorch`.
- neighbour_size: If use dynamic fetching,the number of blocks for each buffer. Note that only the square of an integer N is supported. N^2^ = 9 and N^2^ = 16 have been tested in 1K model.Larger N is supported and N = 1 is illegal for multiblock-based dynamic fetching.

### Set Environment Variables
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
Start running command after preparing distributed config file and ckpt model and appoint the corresponding model type for distributed rendering.See `scripts/` for details.
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs --nnodes=number_of_nodes --node_rank=rank_of_node ./renderer.py --model_type=MultiBlockTorch --config=./confs/dist_render_conf/dist_render_city_multi_branch_encodeapp.txt --ckpt=/your_ckpt_model_path
```
