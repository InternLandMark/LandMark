import cv2
import numpy as np
import scipy.signal
import torch
from torch import nn


def mse2psnr_npy(x):
    return -10.0 * np.log(x) / np.log(10.0)


def n_to_reso(n_voxels, bbox):
    """Compute new grid size.
    Args:
        n_voxels (int): The number of voxels
        bbox (torch.Tensor): The representation of Axis Aligned Bounding Box(aabb)
    Returns:
        list: The current grid size
    """
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    # ratio = (xyz_max - xyz_min).max() / (xyz_max - xyz_min).min()
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


class st:
    """Define color"""

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):

    # sigma, dist  [N_rays, N_samples]
    alpha = 1.0 - torch.exp(-sigma * dist)

    tensor = torch.cumprod(
        torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1),
        -1,
    )

    weights = alpha * tensor[:, :-1]  # [N_rays, N_samples]
    ret = tensor[:, -1:]

    return alpha, weights, ret


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    Visualize depth map.

    Args:
        depth (numpy.ndarray): The depth map.
        minmax (list): The min and max depth.

    Returns:
        numpy.ndarray: The visualized depth map.
        list: The min and max depth.
    """
    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = np.clip(x, a_min=0, a_max=1)
    x = x / 1.1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)


__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    import lpips

    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    """
    Calculate the lpips loss

    Args:
        np_gt (numpy.ndarray): The groundtruth rgb.
        np_im (numpy.ndarray): The compared rgb.
        net_name (str): The network name
        device (str): The device on which a tensor is or will be allocated.
    Returns:
        float: The value of lpips loss.
    """
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """
    Calculate the ssim metric
    """

    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    def filt_fn(z):
        return np.stack(
            [convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :]) for i in range(z.shape[-1])],
            -1,
        )

    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


class TVLoss(nn.Module):
    """Calculate the Total Variation Loss"""

    def __init__(self, TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def check_args(args):
    """Check args during initialization."""
    assert (
        sum([args.plane_parallel, args.channel_parallel, args.branch_parallel]) <= 1
    ), "Only one of the channel/plane/block parallel modes can be True"
    if sum([args.plane_parallel, args.channel_parallel, args.branch_parallel]) == 1 or args.DDP:
        assert args.distributed
    plane_division = args.plane_division
    if args.model_parallel_and_DDP:
        assert args.use_preprocessed_data
        assert args.branch_parallel or args.channel_parallel or args.plane_parallel
        if args.plane_parallel or args.branch_parallel:
            assert args.world_size % (plane_division[0] * plane_division[1]) == 0
    else:
        # plane parallel
        if args.plane_parallel or args.branch_parallel:
            assert (
                plane_division[0] * plane_division[1] == args.world_size
            ), "world size is not equal to num of divided planes"

    if args.use_preprocessed_data:
        assert args.add_lpips == -1
        assert args.batch_size % 8192 == 0
