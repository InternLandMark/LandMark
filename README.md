<p align="center">
    <img src="https://img.shields.io/badge/Trainer-Ready-green"/>
    <img src="https://img.shields.io/badge/Renderer-Ready-green"/>
    <img src="https://img.shields.io/badge/Framework-Ready-green"/>
    <img src="https://img.shields.io/badge/Documentation-Preview-purple"/>
    <img src="https://img.shields.io/badge/License-MIT-orange"/>
</p>

<p align="center">
    <picture>
    <img src="https://raw.githubusercontent.com/InternLandMark/LandMark_Documentation/4f09c93cbec0ad50d27ac52f858e7a6c541168d6/pictures/intern_logo.svg" width="350">
    </picture>
</p>

<p align="center">
    <picture>
    <img src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/pictures/logo.png?raw=true" width="650">
    </picture>
</p>

<p align="center"> <font size="4"> 🌏NeRF the globe if you want </font> </p>

<p align="center">
<picture>
    <img src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/pictures/zhujiajiao.gif?raw=true" width="300">
    </picture>
    <picture>
    <img src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/pictures/wukang_noblackscreen.gif?raw=true" width="300">
    </picture>
    <picture>
    <img src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/pictures/xian.gif?raw=true" width="300">
    </picture>
    <picture>
    <img src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/pictures/sanjiantao.gif?raw=true" width="300">
    </picture>
</p>

<p align="center">
    <a href="https://landmark.intern-ai.org.cn/">
    <font size="4">
    🏠HomePage
    </font>
    </a>
    |
    <a href="https://internlandmark.github.io/LandMark_Documentation/">
    <font size="4">
    📑DocumentationSite
    </font>
    </a>
    |
    <a href="https://city-super.github.io/gridnerf/">
    <font size="4">
    ✍️PaperPage
    </font>
    </a>
    |
    <a href="https://city-super.github.io/matrixcity/">
    <font size="4">
    🏬MatrixCity
    </font>
    </a>
</p>

# 💻 About
This repository contains the source code for the project LandMark, the groundbreaking large-scale 3D real-world city scene modeling and rendering system.<br>
The project is built upon GridNeRF (CVPR23). Please refer to the paper and project page for more details.<br>
Extending from GridNeRF, LandMark drastically improves training and rendering efficiency with parallelization, operators and kernels, as well as a polish over the algorithm.<br>
Including:

- Large-scale, high-quality novel view rendering:
    - For the first time, we realized efficient training of 3D neural scenes on over 100 square kilometers of city data; and the rendering resolution reached 4K. We used over 200 billion learnable parameters to model the scene.
- Multiple feature extensions:
    - Beyond rendering, we showcased layout adjustment such as removing or adding a building, and scene stylization with alternative appearance such as changes of lighting and seasons.
- Training, rendering integrated system:
    - We delivered a system covering algorithms, operators, computing systems, which serves as a solid foundation for the training, rendering and application of real-world 3D large models.
- Distributed rendering system:
    - For real-time rendering of large scale GridNeRF model.

And now it's possible to train and render with your own LandMark models and enjoy your creativity.<br>
Your likes and contributions to the community are exactly what we need.
# 🎨 Support Features
The LandMark supports plenty of features at present:

- GridNeRF Sequential Model Training
- GridNeRF Parallel Model Training
    - Branch Parallel
    - Plane Parallel
    - Channel Parallel
- GridNeRF Sequential Model Rendering
- Pytorch DDP both on training and rendering

- MatrixCity Datasets Supports
- Real-time Distributed Rendering System

It's highly recommended to read the [DOCUMENTATION](https://internlandmark.github.io/LandMark_Documentation/) about the implementations of our parallel acceleration strategies.

# 🚀 Quickstart
## Prerequisites
You must have a NVIDIA video card with [CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) installed on the system.<br>
This library has been tested with single and multiple `A100` GPUs.
## Install LandMark
The LandMark repository files contains configuration files to help you create a proper environment
```
git clone https://github.com/InternLandMark/LandMark.git
```
## Create Environment
We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage complicated dependencies:
```
cd LandMark
conda create --name landmark -y python=3.9.16
conda activate landmark
python -m pip install --upgrade pip
```
This library has been tested with version `3.9.16` of Python.
## Pytorch & CUDA
Install pytorch with CUDA using the commands below once and for all:
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
This library has been tested with version `11.6` of CUDA.
## Dependencies
We  provide `requirements.txt` for setting the environment easily.
```
pip install -r requirements.txt
```
## Prepare Dataset
For confidential reason, the native datasets we use as shown above will not be released.<br>
For ideal reproduction result, [MatrixCity](https://city-super.github.io/matrixcity/) dataset is highly recommanded.<br>
We've prepared tuned configuration files and dedicated dataloader for the MatrixCity dataset.<br>
Large scale scenes captured from the real world are most suitable for our method.<br>
We recommend using dataset of a building, a well-known LandMark and even a small town.<br>
Prepare about 250 ~ 300 images of the reconstruction target. Make sure enough overlapping.<br>
Reform your dataset as the following structure:

- your_dataset/
    - images_1/
        - image_0.png
        - image_1.png
        - image_2.png
        - ...

Folder name `images_1/` indicates that the downsample factor is `1` for the images.<br>
Extracting poses and `sparse` point-cloud model using [COLMAP](https://colmap.github.io/) as other NeRF methods.
Then transfer the poses data using commands below:
```
python app/tools/colmap2nerf.py --recon_dir data/your_dataset/sparse/0 --output_dir data/your_dataset
```
A `transforms_train.json` and a `transforms_test.json` files will be generated in the `your_dataset/` folder.<br>
Referring to the `app/tools/config_parser.py` and the `app/tools/dataloader/city_dataset.py` for help.<br>
## Set Arguments
We provide a configuration file `confs/city.txt` as an example to help you initialize your experiments.<br>
There are bunches of arguments for customization. We divide them into  four types for better understanding<br>
Some important arguments are demonstrated here. Don't forget to specify path-related arguments before proceeding.<br>

- experiment
    - dataroot - Path of the base of datasets. Use `LandMark/datasets` to manage all datasets
    - datadir - Path of your dataset. It's a relative path to the base of datasets
    - dataset_name - Set the type of dataloader rather than the dataset. Using `"city"` as recommended
    - basedir - Where to save your training checkpoint. Using `LandMark/log` by default
- train
    - start_iters - Number of start iteration in training
    - n_iters - Total number of iterations in training
    - batch_size - Training batch size
    - add_nerf - Which iteration to use nerf brunch
- render
    - sampling_opt - Whether to use sampling optimization when rendering
- model
    - resMode - Resolution mode in muti-resolution model

For more details about arguments, refer to the `LandMark/app/config_parser.py`<br>
Tune the `--ub` and `--lb` arguments to achieve ideal result in the experiments.
## Train Model
Now it's time to train your own LandMark model:
```
python app/trainer.py --config confs/city.txt
```
The training checkpoints and images will be saved in `LandMark/log/your_expname` by default.
## Render Images
After the training process completed, independent rendering test is available:
```
python app/renderer.py --config confs/city.txt --ckpt=log/your_expname/your_expname.th
```
The rendering results will be save in `LandMark/log/your_expname/imgs_test_all` by default.
# 📖Learn More
## Directory Structure

- app/
    - models/ - Contains sequential and parallel implementations of GridNeRF models
    - tools/ - Contains dataloaders, train/render utilities
    - tests/ - Contains scripts for integerity test
    - trainer.py - Manage running process for training
    - renderer.py - Manage running process for training
- confs/ - Contains configuration files for experiments
- dist_renders - Code, introduction, scripts about distributed rendering system
- requirements.txt - Environment configuration file for pip

## Pytorch Distributed Data Parallel Support
The trainer and the renderer both support pytorch DDP.<br>
To train with DDP, use commands below:
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs app/trainer.py --config confs/city.txt
```
To render with DDP, use commands below:
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs app/renderer.py --config confs/city.txt --ckpt=log/your_expname/your_expname.th
```
Some arguments related to the multi-GPU environment might need to be set properly. Specify `number_of_GPUs` according to your actual environment.
## Train with the LandMark Parallel Methods
Three types of Parallel strategies are currently supported for training.<br>
It is worth pointing out that all these strategies are adapted for large-scale scene reconstruction with over `2000` images and area of `several acres`<br>
To involve these parallel features in your experiments, simply use the configuration files such as `confs/city_multi_branch_parallel.txt`.<br>
After changing the path arguments in the configuration file, you are ready to train a plug-and-play branch Parallel model:
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs app/trainer.py --config confs/city_multi_branch_parallel.txt
```
There are few differences in use between training a branch parallel model and a sequential model with DDP, but the training efficiency meet great acceleration.<br>
Especially in reconstruction tasks of large scale scenes, our Parallel strategies shows stable adaption of capability in accelerating the whole training process.<br>
To render with the Parallel model after training, using the command as the sequential one
```
python app/renderer.py --config confs/city_multi_branch_parallel.txt --ckpt=log/your_expname/your_expname.th
```
## MatrixCity Dataset
Fully supports the brilliant MatrixCity dataset. Dedicated files are given for block_1 and block_2 in `confs/matrixcity`. <br>
It's recommended to download the datases from [OpenXLab](https://openxlab.org.cn/datasets/bdaibdai/MatrixCity).<br>
For single GPU trainng, simply use:
```
python app/trainer.py --config confs/matrixcity_2block_multi.txt
```
For multi GPU DDP training:
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs app/trainer.py --config confs/matrixcity_2block_multi.txt
```
Other parallel methods are also available:
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs app/trainer.py --config confs/matrixcity_2block_multi_branch_parallel.txt
```
## Real-Time Distributed Rendering System
Large-scale real-time distributed rendering system which supports over 100 km^2 scene and over 30 frames per second. Read [dist_render/README.md](./dist_render/README.md) for more detailes
# 🤝 Authors
The main work comes from the LandMark Team, Shanghai AI Laboratory.<br>

<img src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/pictures/shailab_logo2.jpg?raw=true" width="450">

Here are our honorable Contributors:

<a href="https://github.com/InternLandMark/LandMark/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=InternLandMark/LandMark" />
</a>
