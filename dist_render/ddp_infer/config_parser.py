from app.tools.config_parser import ArgsParser
from app.tools.configs import ArgsConfig


def parse_nerf_config_args(cmd=None):
    """
    parse nerf rendering args
    """
    args_parser = ArgsParser(cmd=cmd)
    exp_args = args_parser.get_exp_args()
    model_args = args_parser.get_model_args()
    render_args = args_parser.get_render_args()
    return ArgsConfig([exp_args, model_args, render_args])
