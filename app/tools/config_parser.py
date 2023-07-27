import configargparse


class ArgsParser:
    """
    ArgsParser is used to parse configures to args.

    Args:
        cmd (str): input arguments
    """

    def __init__(self, cmd=None):
        self.parse_model_args(cmd)
        self.parse_train_args(cmd)
        self.parse_render_args(cmd)
        self.parse_exp_args(cmd)

    def parse_model_args(self, cmd=None):
        """
        Parse model arguments.
        """
        parser = configargparse.ArgumentParser()
        parser.add_argument("--config", is_config_file=True, help="path of config file")
        parser.add_argument("--model_name", type=str, default="GridNeRF")
        parser.add_argument("--resMode", type=int, action="append", help="resolution mode in muti-resolution model")
        parser.add_argument("--run_nerf", default=False, action="store_true", help="whether to use NeRF branch")
        parser.add_argument("--nerf_D", type=int, default=6, help="density-related depth of MLP in NeRF branch")
        parser.add_argument("--nerf_D_a", type=int, default=2, help="color-related depth of MLP in NeRF branch")
        parser.add_argument("--nerf_W", type=int, default=128, help="width of MLP in NeRF branch")
        parser.add_argument(
            "--nerf_freq",
            type=int,
            default=16,
            help="number of channels in frequency encoding for position input in NeRF branch",
        )

        parser.add_argument(
            "--n_importance",
            type=int,
            default=128,
            help="number of sample points along a ray in total. Only work when sampling_opt is true",
        )
        parser.add_argument(
            "--nerf_n_importance", type=int, default=16, help="number of sample points along a ray in NeRF branch"
        )
        parser.add_argument("--nonlinear_density", default=False, action="store_true")

        # volume options
        parser.add_argument(
            "--n_lamb_sigma", type=int, action="append", help="the basis number of density and appearance"
        )
        parser.add_argument("--n_lamb_sh", type=int, action="append", help="the basis number of density and appearance")
        parser.add_argument(
            "--data_dim_color", type=int, default=27, help="in channel dimension of mlp for calculating appearance"
        )

        # network decoder
        parser.add_argument("--pos_pe", type=int, default=6, help="number of pe for pos")
        parser.add_argument("--view_pe", type=int, default=6, help="number of pe for view")
        parser.add_argument("--fea_pe", type=int, default=6, help="number of pe for features")
        parser.add_argument("--featureC", type=int, default=128, help="hidden feature channel in MLP")
        parser.add_argument("--bias_enable", type=int, default=0, help="control the bias of app MLP")

        parser.add_argument(
            "--fea2denseAct",
            type=str,
            default="softplus",
            help="function for feature2density; only support softplus/relu",
        )
        parser.add_argument(
            "--density_shift",
            type=float,
            default=-10,
            help="shift density in softplus; making density = 0  when feature == 0",
        )

        parser.add_argument(
            "--ndims", type=int, default=1, help="the number of dimentions in density and appearance feature matrix"
        )
        if cmd is not None:
            self.model_args = parser.parse_known_args(cmd)[0]
        else:
            self.model_args = parser.parse_known_args()[0]

    def parse_train_args(self, cmd=None):
        """
        Parse train arguments.
        """
        parser = configargparse.ArgumentParser()
        parser.add_argument("--config", is_config_file=True, help="path of config file")
        parser.add_argument("--start_iters", type=int, default=0, help="number of start iteration in training")
        parser.add_argument("--n_iters", type=int, default=30000, help="total number of iterations in training")
        parser.add_argument("--batch_size", type=int, default=4096, help="training batch size")
        parser.add_argument(
            "--progress_refresh_rate",
            type=int,
            default=10,
            help="number of iterations to show psnr or current iteration",
        )
        parser.add_argument(
            "--add_upsample", type=int, default=-1, help="iteration to upsample the training dataset by X2"
        )
        parser.add_argument(
            "--add_lpips",
            type=int,
            default=-1,
            help="iteration to reformat dataset with patch samples and start using lpips loss",
        )
        parser.add_argument("--add_nerf", type=int, default=-1, help="iteration to start using NeRF branch")
        parser.add_argument("--add_distort", type=int, default=-1, help="iteration to start using distortion loss")
        parser.add_argument("--patch_size", type=int, default=128, help="patch size used in lpips")
        parser.add_argument(
            "--residnerf", default=False, action="store_true", help="whether to use residual NeRF or normal NeRF"
        )
        parser.add_argument("--preload", action="store_true", help="whether to preload to cuda")

        parser.add_argument(
            "--train_near_far", type=float, action="append", help="near and far plane along the sample ray in training"
        )

        parser.add_argument("--lr_init", type=float, default=0.02, help="initial learning rate of grid branch")
        parser.add_argument("--lr_basis", type=float, default=1e-3, help="initial learning rate of NeRF branch")
        parser.add_argument(
            "--lr_decay_iters", type=int, default=-1, help="number of iterations for learning rate to decay"
        )
        parser.add_argument(
            "--lr_decay_target_ratio", type=float, default=0.1, help="the target decay ratio of learning rate"
        )
        parser.add_argument(
            "--lr_upsample_reset",
            default=True,
            action="store_true",
            help="whether to reset learning rate to the inital after upsampling",
        )
        parser.add_argument("--L1_weight_inital", type=float, default=0.0, help="initial weight of L1 loss")
        parser.add_argument("--Ortho_weight", type=float, default=0.0, help="weight of orthogonal projection loss")
        parser.add_argument("--TV_weight_density", type=float, default=0.0, help="weight of TV loss in desity")
        parser.add_argument("--TV_weight_app", type=float, default=0.0, help="weight of TV loss in appreance")

        parser.add_argument(
            "--alpha_mask_thre",
            type=float,
            default=0.0001,
            help="threshold for creating alpha mask volume",
        )
        parser.add_argument(
            "--nSamples",
            type=int,
            default=1e6,
            help="sample point each ray, pass 1e6 if automatic adjust",
        )
        parser.add_argument(
            "--step_ratio", type=float, default=0.5, help="how many grids to walk in one step during sampling"
        )

        parser.add_argument("--N_voxel_init", type=int, default=100**3, help="the initial grid size")
        parser.add_argument("--N_voxel_final", type=int, default=300**3, help="the final grid size")
        parser.add_argument("--upsamp_list", type=int, action="append", help="do upsample in specific iters")
        parser.add_argument(
            "--update_AlphaMask_list", type=int, action="append", help="update alphamask in specific iters"
        )
        parser.add_argument("--alpha_grid_reso", type=int, default=256**3, help="max alpha grid resolution")

        parser.add_argument("--wandb", default=False, action="store_true")

        parser.add_argument("--plane_parallel", default=False, action="store_true", help="training in plane parallel")
        parser.add_argument(
            "--channel_parallel", default=False, action="store_true", help="training in channel parallel"
        )
        parser.add_argument("--branch_parallel", default=False, action="store_true", help="training in block parallel")
        parser.add_argument(
            "--model_parallel_and_DDP", default=False, action="store_true", help="enable model parallel and DDP"
        )

        parser.add_argument("--plane_division", type=int, action="append")

        if cmd is not None:
            self.train_args = parser.parse_known_args(cmd)[0]
        else:
            self.train_args = parser.parse_known_args()[0]

    def parse_render_args(self, cmd=None):
        """
        Parse render arguments.
        """
        parser = configargparse.ArgumentParser()
        parser.add_argument("--config", is_config_file=True, help="path of config file")
        parser.add_argument("--render_px", type=int, default=720, help="width of the images in rendering")
        parser.add_argument("--render_fov", type=float, default=65.0, help="fov of the images in rendering")
        parser.add_argument("--render_nframes", type=int, default=100, help="base number of images to render")
        parser.add_argument("--render_skip", type=int, default=1, help="division number to caculate the actual frames")
        parser.add_argument("--render_fps", type=int, default=30, help="fps of the video to render")
        parser.add_argument(
            "--render_spherical",
            default=False,
            action="store_true",
            help="whether to use a sepherical path in rendering",
        )
        parser.add_argument(
            "--render_spherical_zdiff", type=float, default=1.0, help="pose z shift along the sepherical path"
        )
        parser.add_argument("--render_spherical_radius", type=float, default=4.0, help="radius of the spherical path")
        parser.add_argument(
            "--render_downward", type=float, default=-45.0, help="downward angle of the pose along spherical path"
        )
        parser.add_argument(
            "--render_ncircle", type=float, default=1, help="number of runs in sepherical path in rendering"
        )

        parser.add_argument("--render_path", default=False, action="store_true", help="render a specific path")
        parser.add_argument("--render_pathid", type=int, default=0, help="path id in prepared path file")

        parser.add_argument(
            "--render_near_far",
            type=float,
            action="append",
            help="near and far plane along the sample ray in rendering",
        )

        parser.add_argument(
            "--render_lb", type=float, action="append", help="horizontal bound of bounding box in rendering"
        )
        parser.add_argument(
            "--render_ub", type=float, action="append", help="vertical bound of bounding box in rendering"
        )

        parser.add_argument("--render_batch_size", type=int, default=8192, help="renderding batch size")

        parser.add_argument(
            "--distance_scale",
            type=float,
            default=25,
            help="scaling sampling distance for computation",
        )

        parser.add_argument(
            "--compute_extra_metrics",
            type=int,
            default=1,
            help="whether to compute lpips metric",
        )

        parser.add_argument("--generate_videos", type=int, default=0, help="whether to generate videos after rendering")

        parser.add_argument(
            "--sampling_opt",
            default=False,
            action="store_true",
            help="whether to open sampling optimization when rendering",
        )

        # blender flags
        parser.add_argument(
            "--white_bkgd",
            action="store_true",
            help="set to render synthetic data on a white bkgd (always use for dvoxels)",
        )

        # checkpoint type
        parser.add_argument(
            "--ckpt_type",
            type=str,
            default="full",
            help=(
                "loaded checkpoint type of branch parallel when rendering.\nfull: ckpt is fully merged (default); part:"
                " ckpt is a part of fully-merged ckpt; sub: ckpt is not merged."
            ),
        )
        parser.add_argument("--branch_parallel", default=False, action="store_true")

        parser.add_argument("--plane_division", type=int, action="append")

        if cmd is not None:
            self.render_args = parser.parse_known_args(cmd)[0]
        else:
            self.render_args = parser.parse_known_args()[0]

    def parse_exp_args(self, cmd=None):
        """
        Parse experimental arguments.
        """
        parser = configargparse.ArgumentParser()
        parser.add_argument("--config", is_config_file=True, help="path of config file")
        parser.add_argument("--expname", type=str, help="name of current training/rendering experiment")
        parser.add_argument("--basedir", type=str, default="./log", help="path to store the checkpoints and logs")
        parser.add_argument("--partition", type=str, default="all", help="partition name in dataset")
        parser.add_argument("--add_timestamp", type=int, default=0, help="")
        parser.add_argument("--dataroot", type=str, help="root directory of dataset")
        parser.add_argument("--datadir", type=str, default="_", help="data directory of dataset")
        parser.add_argument("--subfolder", type=str, action="append", help="data directory of dataset")
        parser.add_argument("--dataset_name", type=str, default="blender", help="specity dataset name to")
        parser.add_argument("--use_preprocessed_data", action="store_true", help="whether to use preprocessed data")
        parser.add_argument("--processed_data_type", type=str, default="file", help="type of nprocessed data")
        parser.add_argument(
            "--preprocessed_dir",
            type=str,
            default="/cpfs01/shared/landmarks-viewer/preprocessed_data/",
            help="direction of preprocessed dataset",
        )
        parser.add_argument("--camera", type=str, default="normal", help="define the camera")
        parser.add_argument("--lb", type=float, action="append", help="horizontal bound of bounding box")
        parser.add_argument("--ub", type=float, action="append", help="vertical bound of bounding box")

        parser.add_argument(
            "--ckpt",
            type=str,
            default=None,
            help="specific weights npy file to reload for coarse network",
        )

        parser.add_argument(
            "--random_seed", type=int, default=20211202, help="random seed to initial training and rendering"
        )
        parser.add_argument("--debug", action="store_true", help="only for dataset debug")
        parser.add_argument("--local_rank", type=int, default=0, help="used for single machine")

        # logging/saving options
        parser.add_argument("--N_vis", type=int, default=5, help="N images to visualize")
        parser.add_argument("--vis_every", type=int, default=10000, help="frequency of visualize the image")

        parser.add_argument("--filter_ray", default=False, action="store_true", help="whether to prefilter images")

        parser.add_argument(
            "--downsample_train", type=int, default=20, help="multiplier to downsample dataset in training"
        )

        parser.add_argument(
            "--env",
            type=str,
            default="aliyun",
            help="configure running env. Only suppport single_node/slurm/aliyun for now",
        )

        if cmd is not None:
            self.exp_args = parser.parse_known_args(cmd)[0]
        else:
            self.exp_args = parser.parse_known_args()[0]

    def get_model_args(self):
        return self.model_args

    def get_train_args(self):
        return self.train_args

    def get_render_args(self):
        return self.render_args

    def get_exp_args(self):
        return self.exp_args
