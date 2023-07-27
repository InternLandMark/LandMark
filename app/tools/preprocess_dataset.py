import os

from tools.config_parser import ArgsParser
from tools.configs import ArgsConfig
from tools.dataloader import dataset_dict
from tools.slurm import init_distributed_mode

if __name__ == "__main__":

    args_parser = ArgsParser()
    exp_args = args_parser.get_exp_args()
    model_args = args_parser.get_model_args()
    render_args = args_parser.get_render_args()
    train_args = args_parser.get_train_args()

    args = ArgsConfig([exp_args, model_args, render_args, train_args])
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    if args.distributed:
        init_distributed_mode(args)
        dataset = dataset_dict[args.dataset_name]
        train_dataset = dataset(
            split="train", downsample=args.downsample_train, is_stack=args.lpips, args=args, preprocess=True
        )

        if args.processed_data_type == "ceph":
            from petrel_client.client import Client

            conf_path = "~/petreloss.conf"
            client = Client(conf_path)
            train_dataset.distpreprocess(args=args, batch_size=8192, client=client)
        else:
            train_dataset.distpreprocess(args=args, batch_size=8192)
    else:
        dataset = dataset_dict[args.dataset_name]
        train_dataset = dataset(
            split="train", downsample=args.downsample_train, is_stack=args.lpips, args=args, preprocess=True
        )

        if args.processed_data_type == "ceph":
            from petrel_client.client import Client

            conf_path = "~/petreloss.conf"
            client = Client(conf_path)
            train_dataset.preprocess(args=args, split_num=200, batch_size=8192, client=client)
        else:
            train_dataset.preprocess(args=args, split_num=200, batch_size=8192)
