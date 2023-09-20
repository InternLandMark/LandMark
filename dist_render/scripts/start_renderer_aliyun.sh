export PROFILE_STAGES=False
export RENDER_1080P=False
export HALF_PRECISION_INFERENCE=False
export PROFILE_TIMELINE=False
# export TENSOR_PARALLEL_SIZE=4
export HOME=/your_home
export PYTHONPATH=/your_landmark_path/:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 ../renderer.py --model_type=MultiBlockTorch --pose_num=20 --config=/your_dist_config_path --ckpt=/your_ckpt_model_path --save_png
